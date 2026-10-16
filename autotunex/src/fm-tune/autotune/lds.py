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

# LDS based searcher for AutoTune hyperparameter optimization

import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

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


class IncrementalLimitedDiscrepancySearch:
    """
    Limited discrepancy search.
    """

    def __init__(
        self,
        max_discrepancy: int = 1,
        variables: List = None,
        values: List = None,
        verbose: int = 0,
        random_state: int = 42,
        default_values: Optional[Dict] = None,
    ):
        """
        Constructor for the LDS incremental searcher. LDS assumes a discrete
        search space, namely the variables have discrete and finite domains.

        Args:
            max_discrepancy: int
                Maximum discrepancy used for search (default is 1).
            variables: List
                The list of variables defining the search space.
            values: List
                Domains of values for the variables.
            verbose: int
                The verbosity level (default is 0).
            random_state: int
                An integer used for initializing the random number generator.
            default_values: Dict
                A dict with the default valurs of the variables.
        """

        self.max_discrepancy = max_discrepancy
        self.variables = variables  # variable names
        self.values = values  # list of values for each variable
        self.domains = [len(d) for d in self.values]  # variable domains
        self.default_values = default_values  # default values for variables (if any)
        self.verbose = verbose
        self.random_state = random_state

        self.rng = init_random(self.random_state)  # create and seed the RNG (numpy)
        self.configs = []

        if self.default_values:
            self.init_config = []
            for pos, var in enumerate(self.variables):
                var_name = var.split("/")[-1]
                self.init_config.append(self.values[pos].index(self.default_values[var_name]))
        else:
            self.init_config = [self.rng.integers(low=0, high=d) for d in self.domains]

        config = [self.values[x][y] for x, y in enumerate(self.init_config)]
        # self.configs.append(dict(zip(self.variables, config)))
        logger.info(f"[AutoTune] Initialize Incremental LDS with {len(variables)} variables")
        logger.info(f"[AutoTune] Incremental LDS variables: {variables}")
        logger.info(f"[AutoTune] Incremental LDS values: {values}")
        logger.info(f"[AutoTune] Max discrepancy is: {self.max_discrepancy}")
        logger.info(f"[AutoTune] Default values: {self.default_values}")
        logger.info(f"[AutoTune] Initial configuration: {self.init_config} -> {config}")
        print(f"[AutoTune] Initialize Incremental LDS with {len(variables)} variables")
        print(f"[AutoTune] Incremental LDS variables: {variables}")
        print(f"[AutoTune] Incremental LDS values: {values}")
        print(f"[AutoTune] Max discrepancy is: {self.max_discrepancy}")
        print(f"[AutoTune] Default values: {self.default_values}")
        print(f"[AutoTune] Initial configuration: {self.init_config} -> {config}")

        # Initialize the search space
        self.stack = deque()
        root = (-1, [], self.max_discrepancy)
        self.stack.append(root)

    def get_init_config(self):
        """
        Return the initial (default) config.
        """
        config = [self.values[x][y] for x, y in enumerate(self.init_config)]
        result = dict(zip(self.variables, config))
        return result

    def next_config(self):
        """
        Return the next configuration within max discrepancy. If the search space
        has been exhausted then return None.
        """

        def next_node():
            n = None
            if len(self.stack) > 0:
                n = self.stack.pop()
            return n

        # --
        def expand_node(n):
            i, a, k = n[0], n[1], n[2]
            if i >= len(self.variables) - 1:
                return True  # leaf node (full configuration)
            else:
                d = self.domains[i + 1]
                for val in range(d):
                    if val != self.init_config[i + 1]:
                        ch = (i + 1, a + [val], k - 1)
                    else:
                        ch = (i + 1, a + [val], k)
                    if ch[2] >= 0:  # check if discrepancy ok
                        self.stack.append(ch)
                return False

        # --

        n = next_node()
        while n:
            if expand_node(n):
                config = [self.values[x][y] for x, y in enumerate(n[1])]
                result = dict(zip(self.variables, config))
                self.configs.append(result)
                return result
            n = next_node()
        return None


# --


class LimitedDiscrepancySearch(Searcher):
    """
    Distributed Limited Discrepancy Searcher for ray/tune.
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
        system_deadline=0,
    ):
        """
        Construct the Distributed Limited Discrepancy Searcher.

        Args:
            space: Dict
                A dict representing the search space.
            metric:
                The metric used as objective function.
            mode:
                The sense of the optimization (`min` or `max`).
            random_state: int
                An integer used for initializing the randon number generator.
            verbose: int
                The verbosity level (default is 0).
            max_discrepancy: int
                The maximum discrepancy value used during search.
            num_samples: int
                The maximum number of sample configurations to be generated.
            default_values: Dict
                A dict containing the default values of the variables.
            system_deadline: int
                A time deadline in seconds for the search (default is 0 - no timeout).
        """

        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'."

        self._config_counter = defaultdict(int)
        self._default_values = default_values
        self._system_deadline = system_deadline

        super(LimitedDiscrepancySearch, self).__init__(metric=metric, mode=mode)

        if mode == "max":
            self._metric_op = 1.0
        elif mode == "min":
            self._metric_op = -1.0

        self._space = {}
        self._live_trial_mapping = {}
        self._buffered_trial_results = []
        self._max_discrepancy = max_discrepancy
        self._verbose = verbose
        self._random_state = random_state
        self._num_samples = num_samples  # if None then unlimited
        self.num_configs = 0
        self.optimizer = None
        self.first_suggest = True
        self.rstate = np.random.RandomState(random_state)

        # Create the LDS search space and optimizer
        if isinstance(space, dict) and space:
            self._create_search_space(space)
            self._setup_optimizer()

    def _create_search_space(self, space: Dict):
        """
        Create the search space for the searcher.

        Args:
            space: Dict
                A dict containing the hyperparameter search space. All variables
                are assumed to be discrete i.e., with finite domains of values.
        """

        spec = flatten_dict(space, prevent_delimiter=True)
        resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)

        if len(grid_vars) > 0:
            raise ValueError("LDS cannot process grid/continous variables.")

        for path, domain in domain_vars:
            par = "/".join(str(p) for p in path)
            assert isinstance(domain, Categorical)
            values = domain.categories
            self._space[par] = values

        # Set the variables and their domains (for search)
        _vars, _vals = zip(*self._space.items())
        self._variables = list(_vars)  # list of vars (e.g., ['X', 'Y', 'Z])
        self._values = list(_vals)  # list of domains (domain = list of values)

        # Correct the max discrepancy value (if required)
        if self._max_discrepancy == 0:
            logger.info("[AutoTune] Max discrepancy cannot be 0, setting to 1")
            self._max_discrepancy = 1
        if self._max_discrepancy >= len(self._variables):
            logger.info("[AutoTune] Max discrepancy cannot exceed the number of variables")
            self._max_discrepancy = len(self._variables)

        # Unflatten the resolved vars dict (if any)
        if len(resolved_vars) > 0:
            dt = {"/".join(str(p) for p in path): val for path, val in resolved_vars}
            self._resolved_vars = unflatten_list_dict(dt)
        else:
            self._resolved_vars = {}

    def _setup_optimizer(self):
        """
        Initial setup of the LDS searcher.
        """

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        self.optimizer = IncrementalLimitedDiscrepancySearch(
            max_discrepancy=self._max_discrepancy,
            variables=self._variables,
            values=self._values,
            verbose=self._verbose,
            random_state=self._random_state,
            default_values=self._default_values,
        )

    def set_search_properties(self, metric: Optional[str], mode: Optional[str], config: Dict, **spec) -> bool:
        """
        Set the search properties (overrides default implementation).

        Args:
            metric: str
                The metric used as objective function.
            mode: str
                The sense of optimization (min or max).
            config: Dict
                A dict containing the search space.

        Returns:
            `True` if successful and `False` otherwise.
        """

        if self.optimizer:
            return False

        # space = self.convert_search_space(config)
        # self._space = space
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

    def suggest(self, trial_id: str) -> Optional[Dict]:
        """
        Return new point to be explored by black box function.

        Args:
            trial_id (str): Id of the trial.
                This is a short alphanumerical string.

        Returns:
            Either a dictionary describing the new point to explore or
            None, when no new point is to be explored for the time being.
        """
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
            # We return the default config (initial)
            config = self.optimizer.get_init_config()
            self.first_suggest = False
        else:
            # We compute the new point to explore
            config = self.optimizer.next_config()

        self.num_configs += 1
        if config is None:
            return Searcher.FINISHED
        if self._num_samples is not None and self.num_configs > self._num_samples:
            return Searcher.FINISHED

        # Save the new trial to the trial mapping
        self._live_trial_mapping[trial_id] = config
        config = unflatten_dict(config)

        # Return a deep copy of the mapping
        return {**self._resolved_vars, **config}

    def on_trial_complete(self, trial_id: str, result: Optional[Dict] = None, error: bool = False):
        """
        Notification for the completion of trial.

        Args:
            trial_id (str): Id of the trial.
                This is a short alphanumerical string.
            result (dict): Dictionary of result.
                May be none when some error occurs.
            error (bool): Boolean representing a previous error state.
                The result should be None when error is True.
        """
        # We try to get the parameters used for this trial
        params = self._live_trial_mapping.pop(trial_id, None)

        # The results may be None if some exception is raised during the trial.
        # Also, if the parameters are None (were already processed)
        # we interrupt the following procedure.
        # Additionally, if somehow the error is True but
        # the remaining values are not we also block the method
        if result is None or params is None or error:
            return

        # We store the results into a temporary cache
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
            # Backwards compatibility
            self.set_state(save_object)
        else:
            self.rstate.set_state(save_object.pop("__rstate"))
            self.__dict__.update(save_object)


# --

if __name__ == "__main__":
    logger.info("Testing the LimitedDiscrepancySearcher")
    search_space = {
        "A": "fixed",
        "X": tune.choice(["a", "b", "c", "d"]),
        "Y": tune.choice(["a", "b", "c"]),
        "Z": tune.choice(["a", "b", "c"]),
        # "W": tune.choice(["a", "b", "c"]),
        # "V": tune.choice(["a", "b", "c"]),
    }

    default_values = {"X": "a", "Y": "a", "Z": "a"}

    search_algo = LimitedDiscrepancySearch(
        space=search_space, metric="loss", mode="min", max_discrepancy=1, num_samples=10, default_values=default_values
    )

    # search_algo = LimitedDiscrepancySearch(metric="loss", mode="min", max_discrepancy=5)
    search_algo.set_search_properties(metric="loss", mode="min", config=search_space)

    num_configs = 0
    while True:
        cfg = search_algo.suggest(trial_id="tr")
        num_configs += 1
        if cfg == Searcher.FINISHED:
            break
        logger.info(cfg)

    logger.info(f"Generated {num_configs} configurations.")
    logger.info("Done.")
