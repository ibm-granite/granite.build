# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# BLDS based searcher for AutoTune hyperparameter optimization

import json
import logging
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Use cloudpickle instead of pickle to make lambda funcs in HyperOpt pickleable
from ray import cloudpickle, tune
from ray.tune.result import DEFAULT_METRIC
from ray.tune.search import UNDEFINED_METRIC_MODE, UNDEFINED_SEARCH_SPACE, Searcher
from ray.tune.search.sample import Categorical
from ray.tune.search.variant_generator import parse_spec_vars
from ray.tune.utils import flatten_dict
from ray.tune.utils.util import unflatten_dict, unflatten_list_dict

from autotune.utils import init_random

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _dict_hash(config, precision):
    flatconfig = flatten_dict(config)
    for param, value in flatconfig.items():
        if isinstance(value, float):
            flatconfig[param] = "{:.{digits}f}".format(value, digits=precision)

    hashed = json.dumps(flatconfig, sort_keys=True, default=str)
    return hashed


class BLDSBanditArm:
    """
    A bandit arm in BLDS. Each arm corresponds to one pipeline configuration
    and tracks UCB/LCB bounds that tighten as the arm is evaluated at higher
    fidelity rungs.
    """

    def __init__(self, max_size: int):
        """
        Args:
            max_size: int
                The number of fidelity rungs. After `max_size` updates the arm
                is considered finalized: its bounds collapse to the observed
                objective value and it cannot be evaluated again.
        """
        self.lcb = -np.inf
        self.ucb = np.inf
        self.objective = np.inf
        self.step = 0
        self.max_size = max_size
        self.total_data_size = 0.0

    def update(
        self,
        val: float,
        size: float,
        log_flag: bool,
        L: float = 1.0,
        delta: float = 0.1,
        c_const: float = 4.0001,
    ):
        """
        Update the arm with a new observation at the next fidelity rung.

        Args:
            val: float
                The observed objective value (smaller is better).
            size: float
                The number of training samples used at this rung.
            log_flag: bool
                When True, use the paper's log-based exploration term:
                    sd = sqrt(log(c * L * Dk^2 / delta) / Dk)
                When False, fall back to the simpler reference formula:
                    sd = sqrt(0.09 * step / size)
            L: float
                Size of the search space (product of variable domain sizes).
                Only used when log_flag is True.
            delta: float
                Confidence parameter for the PAC-style bound. Only used when
                log_flag is True.
            c_const: float
                Exploration constant from the paper (must be > 4). Only used
                when log_flag is True.
        """
        self.step += 1
        self.objective = val
        self.total_data_size += float(size)

        if log_flag:
            Dk = self.total_data_size
            inner = c_const * L * Dk * Dk / max(delta, 1e-12)
            inner = max(inner, math.e)
            sd = math.sqrt(math.log(inner) / Dk)
        else:
            sd = math.sqrt(0.09 * self.step / max(size, 1.0))

        self.lcb = val - sd
        self.ucb = val + sd

        if self.step >= self.max_size:
            self.lcb = self.objective
            self.ucb = self.objective

    def is_finalized(self) -> bool:
        return self.step >= self.max_size


class IncrementalBanditLDS:
    """
    Concurrent state machine for BLDS.

    Splits BLDS into two layers so that emissions can run ahead of evaluations:
      * Structural layer: given a `pinit`, enumerate all leaves at discrepancy
        <= max_discrepancy into a deque (LDS-style). Independent of observed
        bounds; safe to drain concurrently.
      * Feedback layer: as evaluations complete, update the arm cache and
        re-enqueue arms that fall in the "uncertain" zone vs. the current
        `pinit` for re-evaluation at the next fidelity rung.

    `pinit` replacement is best-effort: when a finalized arm beats `pinit`,
    future enumerations switch to the new `pinit`, but the in-flight stack is
    not invalidated. Pruning still happens at pop time via the arm cache.
    """

    def __init__(
        self,
        max_discrepancy: int = 1,
        variables: List = None,
        values: List = None,
        verbose: int = 0,
        random_state: int = 42,
        default_values: Optional[Dict] = None,
        fidelity_schedule: Optional[List[float]] = None,
        delta: float = 0.1,
        c_const: float = 4.0001,
        log_exploration: bool = True,
    ):
        """
        Args:
            max_discrepancy: int
                Maximum number of variable changes from `pinit` allowed during
                local search around `pinit`.
            variables: List
                The list of variables defining the search space.
            values: List
                Domains of values for the variables.
            verbose: int
                The verbosity level (default is 0).
            random_state: int
                An integer used for initializing the random number generator.
            default_values: Dict
                A dict with the default values of the variables. Used as the
                very first `pinit`; later restarts sample randomly.
            fidelity_schedule: List[float]
                Monotone-increasing list of `hpo_dataset_percentage` rungs in
                the half-open interval (0, 1]. The maximum entry —
                `fidelity_schedule[-1]` — defines the "top rung" and need not
                equal 1.0; whatever it is, that value is the most-expensive
                evaluation tier. The k-th rung evaluates an arm on the first
                `fidelity_schedule[k] * dataset_size` examples.
            delta: float
                Confidence parameter for the PAC-style UCB/LCB bound.
            c_const: float
                Exploration constant from the paper (must be > 4).
            log_exploration: bool
                Use the paper's log-based exploration term in UCB/LCB.
        """

        self.max_discrepancy = max_discrepancy
        self.variables = variables
        self.values = values
        self.domains = [len(d) for d in self.values]
        self.default_values = default_values
        self.verbose = verbose
        self.random_state = random_state

        if fidelity_schedule is None:
            fidelity_schedule = [0.1, 0.25, 0.5, 1.0]
        if not fidelity_schedule:
            raise ValueError("fidelity_schedule must be non-empty.")
        if any(b <= a for a, b in zip(fidelity_schedule, fidelity_schedule[1:])):
            raise ValueError("fidelity_schedule must be strictly monotone-increasing.")
        if any(p <= 0.0 or p > 1.0 for p in fidelity_schedule):
            raise ValueError(f"fidelity_schedule entries must be in (0, 1]; got {fidelity_schedule}")

        self.fidelity_schedule = list(fidelity_schedule)
        self.num_rungs = len(self.fidelity_schedule)
        self.delta = delta
        self.c_const = c_const
        self.log_exploration = log_exploration

        # L = product of domain sizes; appears in the paper's UCB/LCB formula.
        self.search_space_size = 1.0
        for d in self.domains:
            self.search_space_size *= float(d)

        self.rng = init_random(self.random_state)

        # Initial pinit: from default_values if provided, else random.
        if self.default_values:
            init = []
            for pos, var in enumerate(self.variables):
                var_name = var.split("/")[-1]
                init.append(self.values[pos].index(self.default_values[var_name]))
            self.init_config = init
        else:
            self.init_config = [int(self.rng.integers(low=0, high=d)) for d in self.domains]

        self.pinit_indices: Tuple[int, ...] = tuple(self.init_config)

        # Arm cache + bookkeeping.
        self.cache: Dict[Tuple[int, ...], BLDSBanditArm] = {}

        # Concurrent emission queues. Each entry is (indices, target_rung,
        # pinit_lcb_snapshot, pinit_ucb_snapshot). Snapshots are used to skip
        # entries that have become dominated since they were enqueued.
        self.escalation_queue: deque = deque()
        self.structural_stack: deque = deque()

        # Best objective seen so far (lower is better internally).
        self.best_objective = np.inf
        self.best_indices: Optional[Tuple[int, ...]] = None

        # Tracks whether the initial pinit has been emitted at rung 0.
        self.pinit_seeded = False

        # Restart counter for diagnostics.
        self.restart_count = 0

        logger.info(f"[AutoTune] Initialize BLDS with {len(variables)} variables")
        logger.info(f"[AutoTune] BLDS variables: {variables}")
        logger.info(f"[AutoTune] BLDS values: {values}")
        logger.info(f"[AutoTune] BLDS max discrepancy: {self.max_discrepancy}")
        logger.info(f"[AutoTune] BLDS fidelity schedule: {self.fidelity_schedule}")
        logger.info(f"[AutoTune] BLDS delta={self.delta}, c={self.c_const}, log={self.log_exploration}")
        logger.info(f"[AutoTune] BLDS default values: {self.default_values}")
        logger.info(f"[AutoTune] BLDS initial pinit indices: {self.init_config}")

    # -- public API used by the Ray Tune wrapper ------------------------------

    def get_init_config(self) -> Tuple[Dict, float]:
        """
        Return the very first config (default values) at the cheapest rung.
        """
        self.pinit_seeded = True
        config = self._indices_to_config(self.pinit_indices)
        self._enumerate_leaves_around_pinit()
        return config, self.fidelity_schedule[0]

    def next_config(self) -> Optional[Tuple[Dict, float]]:
        """
        Return the next (config, fidelity_pct) to evaluate, or None if the
        search is exhausted.

        Concurrent: never blocks waiting for in-flight reports. If the
        emission queues run dry, restart with a new pinit and re-enumerate.
        """
        # Try up to 2 restart rounds to avoid an infinite loop in edge cases
        # (e.g. a degenerate search space where every restart produces only
        # finalized arms).
        for _ in range(self.max_discrepancy + 2):
            emission = self._pop_next_emission()
            if emission is not None:
                return emission
            if not self._restart():
                return None
        return None

    def report(self, indices: Tuple[int, ...], rung: int, objective: float) -> None:
        """
        Feed back the objective for an emitted arm. Updates the cache, may
        replace `pinit`, and may enqueue the arm for the next fidelity rung.

        Args:
            indices: Tuple[int,...]
                The pipeline-configuration index vector that was evaluated.
            rung: int
                The fidelity rung at which it was evaluated.
            objective: float
                The observed objective (smaller is better; the wrapper
                negates if mode=="max").
        """
        arm = self.cache.get(indices)
        if arm is None:
            arm = BLDSBanditArm(self.num_rungs)
            self.cache[indices] = arm

        # Out-of-order or duplicate report: skip if the arm has already
        # progressed past this rung.
        if rung < arm.step:
            return

        size = self._dataset_count_at_rung(rung)
        arm.update(
            val=objective,
            size=size,
            log_flag=self.log_exploration,
            L=self.search_space_size,
            delta=self.delta,
            c_const=self.c_const,
        )

        if objective < self.best_objective:
            self.best_objective = objective
            self.best_indices = indices

        pinit_arm = self.cache.get(self.pinit_indices)
        if pinit_arm is None:
            # pinit hasn't been evaluated yet (race during startup); nothing
            # to compare against.
            return

        if arm.is_finalized() and arm.objective < pinit_arm.objective:
            # Promote: this arm beats pinit on its full evaluation.
            logger.info(
                f"[AutoTune] BLDS promoting new pinit {indices} "
                f"obj={arm.objective:.6f} (was {self.pinit_indices} obj={pinit_arm.objective:.6f})"
            )
            self.pinit_indices = indices
            # Re-enumerate around the new pinit; old in-flight emissions are
            # filtered against the new bounds at pop time.
            self._enumerate_leaves_around_pinit()
            return

        # If the arm is in the uncertain zone vs pinit and not yet finalized,
        # escalate to the next rung.
        if not arm.is_finalized() and arm.lcb <= pinit_arm.ucb and arm.ucb >= pinit_arm.lcb:
            self.escalation_queue.append((indices, arm.step, pinit_arm.lcb, pinit_arm.ucb))

    # -- internals ------------------------------------------------------------

    def _dataset_count_at_rung(self, rung: int) -> float:
        """
        How many training samples are *added* at this rung relative to the
        previous one. The arm tracks cumulative `total_data_size` internally.

        We report a relative count (a unit dataset has size 1.0); the absolute
        sample count cancels out in the UCB/LCB formula because it appears
        both inside and outside the log.
        """
        if rung == 0:
            return self.fidelity_schedule[0]
        return self.fidelity_schedule[rung] - self.fidelity_schedule[rung - 1]

    def _indices_to_config(self, indices: Tuple[int, ...]) -> Dict:
        config_values = [self.values[x][y] for x, y in enumerate(indices)]
        return dict(zip(self.variables, config_values))

    def _hamming(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        return sum(1 for x, y in zip(a, b) if x != y)

    def _enumerate_leaves_around_pinit(self) -> None:
        """
        Push every leaf at Hamming distance <= max_discrepancy from
        pinit_indices onto the structural stack, snapshotting the current
        pinit's LCB/UCB at enqueue time.

        Mirrors the deque expansion pattern in
        IncrementalLimitedDiscrepancySearch.next_config (autotune/lds.py:137-157).
        """
        pinit_arm = self.cache.get(self.pinit_indices)
        snap_lcb = pinit_arm.lcb if pinit_arm is not None else -np.inf
        snap_ucb = pinit_arm.ucb if pinit_arm is not None else np.inf

        # Frame: (depth, partial_indices, remaining_discrepancy)
        # When depth == n_vars - 1 the partial assignment is a complete leaf.
        n = len(self.variables)
        scratch: deque = deque()
        scratch.append((-1, [], self.max_discrepancy))

        while scratch:
            i, partial, k = scratch.pop()
            if i >= n - 1:
                leaf = tuple(partial)
                if leaf == self.pinit_indices:
                    # Don't re-enqueue pinit itself; it's already been seeded.
                    continue
                self.structural_stack.append((leaf, snap_lcb, snap_ucb))
                continue
            d = self.domains[i + 1]
            for val in range(d):
                if val == self.pinit_indices[i + 1]:
                    child = (i + 1, partial + [val], k)
                else:
                    child = (i + 1, partial + [val], k - 1)
                if child[2] >= 0:
                    scratch.append(child)

    def _pop_next_emission(self) -> Optional[Tuple[Dict, float]]:
        """
        Try to produce the next (config, fidelity_pct) emission. Returns None
        if both the escalation queue and structural stack are exhausted.
        """
        # Initial pinit seeding.
        if not self.pinit_seeded:
            self.pinit_seeded = True
            return self._indices_to_config(self.pinit_indices), self.fidelity_schedule[0]

        # Drain escalation queue first (continues to refine known arms).
        while self.escalation_queue:
            indices, target_rung, snap_lcb, snap_ucb = self.escalation_queue.popleft()
            if not self._is_emission_useful(indices, target_rung, snap_lcb, snap_ucb):
                continue
            return self._indices_to_config(indices), self.fidelity_schedule[target_rung]

        # Drain structural stack.
        while self.structural_stack:
            indices, snap_lcb, snap_ucb = self.structural_stack.pop()
            arm = self.cache.get(indices)
            target_rung = arm.step if arm is not None else 0
            if not self._is_emission_useful(indices, target_rung, snap_lcb, snap_ucb):
                continue
            return self._indices_to_config(indices), self.fidelity_schedule[target_rung]

        return None

    def _is_emission_useful(
        self,
        indices: Tuple[int, ...],
        target_rung: int,
        snap_lcb: float,
        snap_ucb: float,
    ) -> bool:
        arm = self.cache.get(indices)
        if arm is not None and arm.is_finalized():
            return False
        if arm is not None and target_rung < arm.step:
            # Already evaluated past this rung; skip the duplicate.
            return False
        if target_rung >= self.num_rungs:
            return False
        if arm is not None:
            # If the arm's LCB has risen above the snapshotted pinit UCB, it
            # is dominated even relative to the (older) pinit context that
            # enqueued it. Skip.
            if arm.lcb > snap_ucb:
                return False
        return True

    def _restart(self) -> bool:
        """
        Start a new outer round: pick a fresh `pinit` (random) and
        re-enumerate. Returns True if a new round was seeded, False if the
        search space is exhausted.

        BLDS in the paper restarts indefinitely until the time budget is
        consumed; we let the wrapper's `num_samples` / `system_deadline`
        checks bound the outer loop. The only structural failure mode is a
        degenerate single-config search space, which we detect below.
        """
        # If every reachable arm is finalized, there is nothing left to do.
        # In practice the wrapper terminates on num_samples first.
        if self._all_reachable_arms_finalized():
            return False

        new_pinit = tuple(int(self.rng.integers(low=0, high=d)) for d in self.domains)
        self.pinit_indices = new_pinit
        self.pinit_seeded = False
        self.restart_count += 1
        logger.info(f"[AutoTune] BLDS restart #{self.restart_count} pinit={new_pinit}")
        # Seeding happens lazily on the next _pop_next_emission call.
        return True

    def _all_reachable_arms_finalized(self) -> bool:
        # Cheap bound: only true when every arm in cache is finalized AND
        # the cache covers the entire search space. We don't enumerate the
        # whole space here; the wrapper's num_samples cap bounds the loop.
        if not self.cache:
            return False
        total_configs = int(self.search_space_size)
        if len(self.cache) < total_configs:
            return False
        return all(arm.is_finalized() for arm in self.cache.values())


class BanditLimitedDiscrepancySearch(Searcher):
    """
    Distributed Bandit Limited Discrepancy Searcher for ray/tune.
    """

    def __init__(
        self,
        space: Optional[Dict] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        random_state: int = 42,
        verbose: int = 0,
        max_discrepancy: int = 1,
        num_samples: Optional[int] = None,
        default_values: Optional[Dict] = None,
        fidelity_schedule: Optional[List[float]] = None,
        delta: float = 0.1,
        c_const: float = 4.0001,
        log_exploration: bool = True,
        system_deadline: int = 0,
    ):
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."

        self._default_values = default_values
        self._system_deadline = system_deadline

        super(BanditLimitedDiscrepancySearch, self).__init__(metric=metric, mode=mode)

        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0

        self._space = {}
        self._live_trial_mapping: Dict[str, Tuple[Tuple[int, ...], int]] = {}
        self._buffered_trial_results = []
        self._max_discrepancy = max_discrepancy
        self._verbose = verbose
        self._random_state = random_state
        self._num_samples = num_samples
        self._fidelity_schedule = fidelity_schedule
        self._delta = delta
        self._c_const = c_const
        self._log_exploration = log_exploration

        self.num_configs = 0
        self.optimizer: Optional[IncrementalBanditLDS] = None
        self.first_suggest = True
        self.rstate = np.random.RandomState(random_state)

        if isinstance(space, dict) and space:
            self._create_search_space(space)
            self._setup_optimizer()

    def _create_search_space(self, space: Dict):
        spec = flatten_dict(space, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if len(grid_vars) > 0:
            raise ValueError("BLDS cannot process grid/continuous variables.")

        for path, domain in domain_vars:
            par = "/".join(str(p) for p in path)
            assert isinstance(domain, Categorical)
            values = domain.categories
            self._space[par] = values

        _vars, _vals = zip(*self._space.items())
        self._variables = list(_vars)
        self._values = list(_vals)

        if self._max_discrepancy == 0:
            logger.info("[AutoTune] Max discrepancy cannot be 0, setting to 1")
            self._max_discrepancy = 1
        if self._max_discrepancy >= len(self._variables):
            logger.info("[AutoTune] Max discrepancy cannot exceed the number of variables")
            self._max_discrepancy = len(self._variables)

        if len(resolved_vars) > 0:
            dt = {"/".join(str(p) for p in path): val for path, val in resolved_vars}
            self._resolved_vars = unflatten_list_dict(dt)
        else:
            self._resolved_vars = {}

    def _setup_optimizer(self):
        if self._metric is None and self._mode:
            self._metric = DEFAULT_METRIC

        self.optimizer = IncrementalBanditLDS(
            max_discrepancy=self._max_discrepancy,
            variables=self._variables,
            values=self._values,
            verbose=self._verbose,
            random_state=self._random_state,
            default_values=self._default_values,
            fidelity_schedule=self._fidelity_schedule,
            delta=self._delta,
            c_const=self._c_const,
            log_exploration=self._log_exploration,
        )

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        if self.optimizer:
            return False

        self._create_search_space(config)
        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0

        self._setup_optimizer()
        return True

    def _inject_fidelity(self, config: Dict, fidelity_pct: float) -> Dict:
        """
        Add two BLDS-controlled keys to the flat config:

          - `training_config/hpo_dataset_percentage = fidelity_pct` so the
            driver loads the right prefix slice for this trial.
          - `training_config/_blds_top_rung_pct = max(fidelity_schedule)` so
            drivers can detect "is this the top rung" without needing to
            know the schedule. The leading underscore marks this as an
            internal field that drivers consume but do not pass to the
            trainer.

        Uses the flattened-key convention shared with LDS (and unflattened
        by the caller).
        """
        result = dict(config)
        result["training_config/hpo_dataset_percentage"] = float(fidelity_pct)
        result["training_config/_blds_top_rung_pct"] = float(self.optimizer.fidelity_schedule[-1])
        return result

    def _config_to_indices(self, config: Dict) -> Tuple[int, ...]:
        idx = []
        for var, domain in zip(self._variables, self._values):
            value = config[var]
            idx.append(list(domain).index(value))
        return tuple(idx)

    def suggest(self, trial_id: str) -> Optional[Dict]:
        if not self.optimizer:
            raise RuntimeError(UNDEFINED_SEARCH_SPACE.format(cls=self.__class__.__name__, space="space"))

        if not self._metric or not self._mode:
            raise RuntimeError(
                UNDEFINED_METRIC_MODE.format(cls=self.__class__.__name__, metric=self._metric, mode=self._mode)
            )

        if self._system_deadline > 0 and time.time() > self._system_deadline:
            logger.info(f"[AutoTune] System deadline {self._system_deadline} is reached.")
            return Searcher.FINISHED

        if self.first_suggest:
            emission = self.optimizer.get_init_config()
            self.first_suggest = False
        else:
            emission = self.optimizer.next_config()

        self.num_configs += 1
        if emission is None:
            return Searcher.FINISHED
        if self._num_samples is not None and self.num_configs > self._num_samples:
            return Searcher.FINISHED

        config, fidelity_pct = emission
        indices = self._config_to_indices(config)

        # Find the rung this emission corresponds to (matches the fidelity
        # value to a rung index, using exact equality since the optimizer
        # picks values directly from the schedule).
        rung = self._fidelity_index(fidelity_pct)

        config = self._inject_fidelity(config, fidelity_pct)

        self._live_trial_mapping[trial_id] = (indices, rung)
        config = unflatten_dict(config)

        return {**self._resolved_vars, **config}

    def _fidelity_index(self, fidelity_pct: float) -> int:
        schedule = self.optimizer.fidelity_schedule
        for i, p in enumerate(schedule):
            if abs(p - fidelity_pct) < 1e-9:
                return i
        # Fallback: closest rung. Should not happen in practice.
        return min(range(len(schedule)), key=lambda i: abs(schedule[i] - fidelity_pct))

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        params = self._live_trial_mapping.pop(trial_id, None)

        if params is None:
            return

        indices, rung = params

        if error or result is None:
            # Treat failure as a heavily dominated arm so it is not retried.
            self.optimizer.report(indices, rung, float("inf"))
            return

        raw = result.get(self._metric)
        if raw is None:
            return

        # Defensive guard against scheduler-truncated trials at non-top rungs.
        # fm-tune drivers set result["done"] = True only after natural training
        # completion. ASHA-truncated trials never reach that line (the trial
        # body is killed mid-train), so result["done"] will be missing/False.
        # At non-top rungs, recording a partial-training loss as if it were
        # complete would distort BLDS's UCB/LCB math; skip the update and
        # leave the arm available for re-emission.
        # See plan silky-cantering-lovelace.md (Phase 3 + Open Question #2).
        is_truncated = not bool(result.get("done", False))
        is_top_rung = rung >= self.optimizer.num_rungs - 1
        if is_truncated and not is_top_rung:
            logger.info(
                f"[AutoTune] BLDS skipping cache update for truncated trial "
                f"{trial_id} at rung {rung} (no result['done']=True)"
            )
            return

        # BLDS minimizes internally. If the user requested mode="max", flip.
        objective = float(raw) * (-self._metric_op)

        self.optimizer.report(indices, rung, objective)
        self._buffered_trial_results.append((params, result))

    def get_state(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        return state

    def set_state(self, state: Dict[str, Any]):
        self.__dict__.update(state)

    def save(self, checkpoint_path: str) -> None:
        save_object = self.__dict__.copy()
        save_object["__rstate"] = self.rstate.get_state()
        with open(checkpoint_path, "wb") as f:
            cloudpickle.dump(save_object, f)

    def restore(self, checkpoint_path: str) -> None:
        with open(checkpoint_path, "rb") as f:
            save_object = cloudpickle.load(f)

        if "__rstate" not in save_object:
            self.set_state(save_object)
        else:
            self.rstate.set_state(save_object.pop("__rstate"))
            self.__dict__.update(save_object)


# --

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing BanditLimitedDiscrepancySearch")

    search_space = {
        "A": "fixed",
        "X": tune.choice(["a", "b", "c", "d"]),
        "Y": tune.choice(["a", "b", "c"]),
        "Z": tune.choice(["a", "b", "c"]),
    }
    default_values = {"X": "a", "Y": "a", "Z": "a"}

    target = {"X": "c", "Y": "b", "Z": "a"}

    def fake_loss(cfg: Dict, fidelity_pct: float) -> float:
        # Lower is better. Distance to target plus a fidelity-dependent noise
        # that shrinks at higher rungs.
        cfg = unflatten_dict(cfg) if isinstance(cfg, dict) else cfg
        tc = cfg["training_config"]["hpo_dataset_percentage"]
        flat = {**cfg}
        flat.pop("training_config", None)
        # Hamming-style distance.
        dist = sum(0 if flat.get(k) == v else 1 for k, v in target.items())
        noise = (1.0 - tc) * 0.05
        return float(dist) + noise

    search_algo = BanditLimitedDiscrepancySearch(
        space=search_space,
        metric="loss",
        mode="min",
        max_discrepancy=2,
        num_samples=40,
        default_values=default_values,
        fidelity_schedule=[0.25, 0.5, 1.0],
    )

    # Concurrent property: three suggest() calls before any complete.
    pending = []
    for tid in ("t1", "t2", "t3"):
        cfg = search_algo.suggest(trial_id=tid)
        assert cfg != Searcher.FINISHED, "Searcher finished too early"
        logger.info(f"  initial concurrent suggest tid={tid} -> {cfg}")
        pending.append((tid, cfg))
    distinct = {json.dumps(c, sort_keys=True, default=str) for _, c in pending}
    assert len(distinct) == 3, f"Concurrent suggestions must be distinct, got {distinct}"

    # Drain results and continue.
    for tid, cfg in pending:
        loss = fake_loss(cfg, cfg["training_config"]["hpo_dataset_percentage"])
        search_algo.on_trial_complete(trial_id=tid, result={"loss": loss, "done": True})

    seen = 0
    while True:
        tid = f"t-{seen}"
        cfg = search_algo.suggest(trial_id=tid)
        if cfg == Searcher.FINISHED:
            break
        loss = fake_loss(cfg, cfg["training_config"]["hpo_dataset_percentage"])
        search_algo.on_trial_complete(trial_id=tid, result={"loss": loss, "done": True})
        seen += 1
        if seen > 200:
            break

    logger.info(f"Generated {search_algo.num_configs} configurations.")
    logger.info(f"Best objective seen: {search_algo.optimizer.best_objective}")
    logger.info(f"Best indices: {search_algo.optimizer.best_indices}")
    logger.info(f"Cache size: {len(search_algo.optimizer.cache)}")
    logger.info(f"Restarts: {search_algo.optimizer.restart_count}")
    logger.info("Done.")
