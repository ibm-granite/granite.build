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

import json
import logging
import os
import random
import re
import shutil
import string
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from autotune.catalog import get_autotune_config, get_autotune_dataset_types  # noqa: F401  (back-compat re-export)

try:
    import numpy as np
    import pandas as pd
    import ray
    import torch
    import transformers
    from datasets import DatasetDict, load_dataset
    from peft import (
        LoHaConfig,
        LoKrConfig,
        LoraConfig,
        PeftType,
        TaskType,
        VeraConfig,
        prepare_model_for_kbit_training,
    )
    from ray import tune
    from ray.air import Result
    from ray.tune import ResultGrid, TuneConfig
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
        PreTrainedModel,
        PreTrainedTokenizer,
    )

    # Local imports
    import autotune.constants
    from autotune.constants import AUTOTUNE_DEFAULT_METRIC, AUTOTUNE_DEFAULT_MODE
except ImportError as exc:
    raise ImportError(
        "autotune's training features require the heavy ML stack "
        "(or an installed training dependency failed to import). "
        'Install or repair it with:  pip install -e ".[full]"\n'
    ) from exc

logger = logging.getLogger(__name__)


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> MutableMapping:
    """
    Flatten a nested dictionary structure.

    Args:
        d: MutableMapping
            The input nested dictionary.
        parent_key: str
            The parent key.
        sep: str
            The separator used for the new keys.

    Returns:
        A flatten dictionary.
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_torch_dtype(precision: str):
    """
    Get the correct torch dtype from the presicion string.

    Args:
        precision: str
            The precision used to load the pretrained model. The allowed values
            are "fp32", "bf16", "int4" and "int8".
    """

    if precision == "fp32":
        dtorch_type = torch.float32
    elif precision in ["fp16", "bf16"]:
        dtorch_type = torch.bfloat16
    elif precision in ["int4", "int8"]:
        dtorch_type = torch.int8  # use bitsandbytes config instead
    else:
        dtorch_type = torch.float32

    return dtorch_type


# Create and initialize random number generator (numpy)
def init_random(seed=None):
    return np.random.default_rng(seed)


# --


def get_strategy(value: Dict[str, Any]):
    """
    Get search space strategy from config.
    A search space defines valid values for your hyperparameters and
    can specify how these values are sampled.

    Refer to the documentation for more info:
    https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs

    The user will have to define the search space in the config file by providing
    the name of the `strategy` and the `values` to sample from.

    The valid strategies are:
    - `uniform` (List) - Samples uniformly between the given bounds.
    - `quniform` (List) - Samples uniformly between the given bounds, quantized.
    - `loguniform` (List) - Samples uniformly between the given bounds on a log scale.
    - `qloguniform` (List) - Samples uniformly between the given bounds on a log scale, quantized.
    - `randn` (List) - Samples from a normal distribution.
    - `qrandn` (List) - Samples from a normal distribution, quantized.
    - `randint` (List) - Samples uniformly between the given bounds, quantized to integers.
    - `qrandint` (List) - Samples uniformly between the given bounds, quantized to integers.
    - `lograndint` (List) - Samples uniformly between the given bounds on a log scale, quantized to integers.
    - `qlograndint` (List) - Samples uniformly between the given bounds on a log scale, quantized to integers.
    - `choice` (List) - Samples from a discrete set of values.
    - `qrandn` (List) - Samples from a normal distribution, quantized.
    - `grid_search` (List) - Samples from the given list of values.

    Args:
        value: dict
            The input dict representing the search stratefy in the config file.

    Returns:
        A ray.tune search strategy.
    """

    assert "strategy" in value.keys(), "Strategy not found."
    assert "values" in value.keys(), "Values not found."

    strategy = value["strategy"]
    if strategy == "uniform":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 2
        return tune.uniform(*value["values"])
    elif strategy == "quniform":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 3
        return tune.quniform(*value["values"])
    elif strategy == "loguniform":
        assert isinstance(value["values"], list)
        assert 2 <= len(value["values"]) <= 3
        return tune.loguniform(*value["values"])
    elif strategy == "qloguniform":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 4
        return tune.qloguniform(*value["values"])
    elif strategy == "randn":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 2
        return tune.randn(*value["values"])
    elif strategy == "qrandn":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 3
        return tune.qrandn(*value["values"])
    elif strategy == "randint":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 2
        return tune.randint(*value["values"])
    elif strategy == "qrandint":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 3
        return tune.qrandint(*value["values"])
    elif strategy == "lograndint":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 3
        return tune.lograndint(*value["values"])
    elif strategy == "qlograndint":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 4
        return tune.qlograndint(*value["values"])
    elif strategy == "choice":
        assert isinstance(value["values"], list)
        return tune.choice(value["values"])
    elif strategy == "string":
        assert isinstance(value["values"], list)
        assert len(value["values"]) == 1
        # For string parameters, we must override the values list with the default value
        value["values"] = [value["default"]]
        return tune.choice(value["values"])
    elif strategy == "grid":
        assert isinstance(value["values"], list)
        return tune.grid_search(value["values"])


def get_default_value(value: Dict[str, Any]):
    assert "default" in value.keys(), "Default value not found."
    return value["default"]


def get_tuner_flag(value: Dict[str, Any]):
    assert "for_tuner" in value.keys(), "`for_tuner` value not found."
    return value["for_tuner"]


def get_param_space(tuner_config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Prepare the parameter search space.

    Args:
        config: dict
            The input configuration of the hyperparameter space (tuner config).

    Returns:
        A tuple containing a dict representing the search space, a second
        dict representing the default values of the searchable parameters and
        a third dict containing the `for_tuner` boolean flags.
    """

    if len(tuner_config) == 0:
        return {}, {}, {}

    # Safety checks
    if "hyperparams" not in tuner_config.keys():
        raise ValueError("Hyperparameter section not found in tuner config.")

    search_space = {}
    default_values = {}
    tuner_flags = {}
    for param, value in tuner_config.get("hyperparams").items():
        search_space[param] = get_strategy(value)
        default_values[param] = get_default_value(value)
        tuner_flags[param] = get_tuner_flag(value)

    return search_space, default_values, tuner_flags


def make_param_space(config: dict):
    """
    Make the parameter space from a hyperprameter config.

    Args:
        config: dict
            A dictionary containing the hyperparameter config.

    Returns:
        A tuple containing a dict representing the search space and a second
        dict representing the default values of the searchable parameters.

    """
    default_values = {}
    param_space = {}
    for k, v in config.items():
        if k in ["tune_config", "training_config", "training_rl_config"]:
            continue  # skip
        param_space[k] = tune.choice([v])
        default_values[k] = v

    return param_space, default_values


def filter_config(config: dict, keyword: str):
    """
    Remove the items that do not match the keywork.

    Args:
        config: dict
            A dict representing a configuration of parameters.
        keywork: str
            A string representing a keyword.

    Returns:
        A dict representing the new configuration that contains only the items
        that match the input keyword.
    """

    return dict((k, v) for k, v in config.items() if k.startswith(keyword))


def get_search_alg(tune_config: dict, default_values: dict = None):
    """
    Initialize the search algorithm and return it.

    Args:
        tune_config: dict
            The configuration required by ray.tune.
        default_values: dict
            The default values of the searchable parameters.

    Returns:
        An instance of the search algorithm specified in the input config. The
        supported algorithms are:
            - `blds`: Bandit Limited Discrepancy Search
            - `bohb`: BOHB
            - `hyperopt`: Hyperopt
            - `lds`: Limited Discrepancy Search
            - `random`: Random Search
    """
    search_alg = tune_config["search_alg"]

    if search_alg == "bohb":
        try:
            from ray.tune.search.bohb import TuneBOHB
        except ImportError:
            raise ImportError("Please pip install hpbandster and ConfigSpace to use TuneBOHB.")

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for TuneBOHB."

        return TuneBOHB()
    elif search_alg == "hyperopt":
        try:
            from ray.tune.search.hyperopt import HyperOptSearch
        except ImportError:
            raise ImportError("Please pip install -U hyperopt to use HyperOptSearch.")

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for HyperOptSearch."

        return HyperOptSearch(metric=tune_config["metric"], mode=tune_config["mode"])
    elif search_alg == "random":
        return None
    elif search_alg == "lds":
        try:
            from autotune.lds import LimitedDiscrepancySearch
        except ImportError:
            raise ImportError("Please import the LimitedDiscrepancySearch class.")

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys()
        "Please specify metric and mode for LimitedDiscrepancySearch."

        return LimitedDiscrepancySearch(
            metric=tune_config["metric"],
            mode=tune_config["mode"],
            max_discrepancy=tune_config.pop("max_discrepancy", 1),
            default_values=default_values,
        )
    elif search_alg == "blds":
        try:
            from autotune.blds import BanditLimitedDiscrepancySearch
        except ImportError:
            raise ImportError("Please import the BanditLimitedDiscrepancySearch class.")

        assert "metric" in tune_config.keys() and "mode" in tune_config.keys(), (
            "Please specify metric and mode for BanditLimitedDiscrepancySearch."
        )

        return BanditLimitedDiscrepancySearch(
            metric=tune_config["metric"],
            mode=tune_config["mode"],
            max_discrepancy=tune_config.pop("max_discrepancy", 1),
            default_values=default_values,
            fidelity_schedule=tune_config.pop("fidelity_schedule", [0.1, 0.25, 0.5, 1.0]),
            delta=tune_config.pop("blds_delta", 0.1),
            c_const=tune_config.pop("blds_c", 4.0001),
            log_exploration=tune_config.pop("blds_log_exploration", True),
            num_samples=tune_config.get("num_samples"),
        )

    else:
        NotImplementedError("Search algorithm not supported.")


def get_scheduler(tune_config: dict):
    """
    Initialize the scheduler and return it.

    The schedulers can early terminate bad trials, pause trials,
    clone trials, and alter hyperparameters of a running trial.

    Refer to the documentation for more info:
    https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#tune-schedulers

    Currently available schedulers are:
        - `fifo` - First In First Out scheduler (default).
        - `asha` - Async Successive Halving (recommended early-stopping scheduler).
        - `hyperbandforbohb` - HyperBand variant locked to the `bohb` searcher.

    Args:
        tune_config: dict
            The configuration required by ray.tune.

    Returns:
        An instance of the scheduler specified in the input config.
    """
    scheduler = tune_config["scheduler"]

    if scheduler == "hyperbandforbohb":
        return tune.schedulers.HyperBandForBOHB()
    elif scheduler == "asha":
        from ray.tune.schedulers import ASHAScheduler

        # asha_max_t is auto-derived from training_config["hpo_num_epochs"] in
        # autotune.optimizer._auto_derive_asha_max_t and injected into
        # tune_config before this function runs. There is no YAML knob for it
        # — keeping the cap in lockstep with the controller's
        # training_iteration stop avoids "ASHA cuts before / after the
        # controller does" alignment bugs. The pop-with-None fallback below
        # is a defensive safety net; if it fires, Ray's built-in max_t=100
        # default applies (after omitting the kwarg, since ASHAScheduler
        # asserts max_t > 0 and a None value blows up that assert).
        asha_kwargs = dict(
            time_attr=tune_config.pop("asha_time_attr", "training_iteration"),
            grace_period=tune_config.pop("asha_grace_period", 1),
            reduction_factor=tune_config.pop("asha_reduction_factor", 3),
            brackets=tune_config.pop("asha_brackets", 1),
        )
        asha_max_t = tune_config.pop("asha_max_t", None)
        if asha_max_t is not None:
            asha_kwargs["max_t"] = asha_max_t
        return ASHAScheduler(**asha_kwargs)
    elif scheduler == "fifo":
        return None
    else:
        NotImplementedError("Scheduler not supported.")


# Compatibility matrix derived from per-searcher source-code analysis.
# See plans silky-cantering-lovelace.md (per-searcher ASHA compatibility) and
# sandy-relaxing-vole.md (scheduler trim) for the rationale behind each entry.
# The supported schedulers after the trim are {fifo, asha, hyperbandforbohb}.
#   - random / lds: non-adaptive; ignore result content -> ASHA is safe
#   - blds: adaptive arm cache; the driver-level FinalSaveOnlyReportCallback
#     gating prevents partial-training results from corrupting BLDS bounds
#   - hyperopt: registers every result as a final observation ->
#     ASHA-truncated trials silently corrupt the TPE model
#   - bohb: reads result["hyperband_info"]["budget"] which only HyperBandForBOHB injects
_SEARCH_ALG_SCHEDULER_COMPAT = {
    "random": {"fifo", "asha"},
    "lds": {"fifo", "asha"},
    "blds": {"fifo", "asha"},
    "hyperopt": {"fifo"},
    "bohb": {"hyperbandforbohb"},
}


def validate_search_alg_scheduler_combo(search_alg: Optional[str], scheduler: Optional[str]) -> None:
    """
    Reject incompatible search_alg x scheduler combinations.

    Args:
        search_alg: str or None
            Name of the search algorithm. None defaults to "random".
        scheduler: str or None
            Name of the scheduler. None defaults to "fifo".

    Raises:
        ValueError if the combination is known to corrupt the searcher's
        internal model or is rejected upstream by Ray Tune.
    """
    if search_alg is None:
        search_alg = "random"
    if scheduler is None:
        scheduler = "fifo"

    allowed = _SEARCH_ALG_SCHEDULER_COMPAT.get(search_alg)
    if allowed is None:
        # Unknown searcher; let downstream code raise its own error.
        return
    if scheduler not in allowed:
        raise ValueError(
            f"search_alg={search_alg!r} is not compatible with scheduler={scheduler!r}. "
            f"Allowed schedulers for {search_alg!r}: {sorted(allowed)}."
        )


def get_tune_config(tune_config: Dict[str, Any], default_values: Dict[str, Any] = None):
    """
    Get the tune config to initialized `tune.TuneConfig` to be passed to `tune.Tuner`.

    Args:
        tune_config: Dict[str, Any]
            A dict containing the ray.tune config.
        default_values: Dict[str, Any]
            A dict containing the default values of the searchable parameters.

    Returns:
        A dict required by tune.TuneConfig. The input tune_config is overwritten!
    """

    # Set the metric
    if "metric" not in tune_config.keys():
        tune_config["metric"] = AUTOTUNE_DEFAULT_METRIC

    # Set the mode
    if "mode" not in tune_config.keys():
        tune_config["mode"] = AUTOTUNE_DEFAULT_MODE

    # Validate search_alg x scheduler combo while both are still strings,
    # before they get replaced with searcher/scheduler instances below.
    validate_search_alg_scheduler_combo(
        tune_config.get("search_alg"),
        tune_config.get("scheduler"),
    )

    # Set the search algorithm
    if "search_alg" in tune_config.keys() and tune_config["search_alg"] is not None:
        tune_config["search_alg"] = get_search_alg(tune_config, default_values)

    # Set the scheduler
    if "scheduler" in tune_config.keys() and tune_config["scheduler"] is not None:
        tune_config["scheduler"] = get_scheduler(tune_config)

    # Remove config keys with None values.
    tune_config = {k: v for k, v in tune_config.items() if v is not None}

    # Remove config keys that are not in TuneConfig
    # extra_keys = ['max_discrepancy']
    # tune_config = {k: v for k, v in tune_config.items() if k not in extra_keys}
    tune_config = {k: v for k, v in tune_config.items() if k in TuneConfig.__dict__}

    return tune_config


def load_datasets(train_file: str, eval_file: str = None, test_file: str = None) -> DatasetDict:
    """
    Load the training, validation and test datasets. The supported file formats
    are: CSV, JSON and JSONL (see also Hugging Face's datasets).

    Args:
        train_file: str
            The path to the file containing the training data.
        eval_file: str
            The path to the file containing the validation data.
        test_file: str
            The path to the file containing the test data.

    Returns:
        A DatasetDict dataset representing the data.

    """

    # Sanity checks
    if train_file is None and eval_file is None and test_file is None:
        raise ValueError("Need at least the training, validation or test files.")
    else:
        if train_file is not None:
            extension = train_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "jsonl",
            ], "`train_file` should be a csv or a json file."
        if eval_file is not None:
            extension = eval_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "jsonl",
            ], "`validation_file` should be a csv or a json file."
        if test_file is not None:
            extension = test_file.split(".")[-1]
            assert extension in [
                "csv",
                "json",
                "jsonl",
            ], "`test_file` should be a csv or a json file."

    # Load the local training and validation datasets
    data_files = {}
    if train_file is not None:
        data_files["train"] = train_file
    if eval_file is not None:
        data_files["eval"] = eval_file
    if test_file is not None:
        data_files["test"] = test_file

    extension = "json" if extension.startswith("json") else extension
    raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    logger.info(f"[AutoTune] Loaded raw training file: {train_file}")
    logger.info(f"[AutoTune] Loaded raw validation file: {eval_file}")
    logger.info(f"[AutoTune] Loaded raw test file: {test_file}")
    logger.info(f"[AutoTune] Raw datasets: {raw_datasets}")

    print(f"[AutoTune] Loaded raw training file: {train_file}")
    print(f"[AutoTune] Loaded raw validation file: {eval_file}")
    print(f"[AutoTune] Loaded raw test file: {test_file}")
    print(f"[AutoTune] Raw datasets: {raw_datasets}")

    return raw_datasets


def resolve_trust_remote_code(default: bool = True) -> bool:
    """Whether HF ``from_pretrained`` calls should pass ``trust_remote_code=True``.

    Loading a model, tokenizer, or config with ``trust_remote_code=True`` executes
    arbitrary Python shipped in the (operator-supplied) model repo. fm-tune
    defaults this to ``True`` because Granite hybrid and some custom architectures
    ship their modeling code in the repo. Operators who only load architectures
    bundled with ``transformers`` can harden this by setting the environment
    variable ``FMTUNE_TRUST_REMOTE_CODE=0`` (also accepts ``false`` / ``no`` / ``off``).
    """
    val = os.getenv("FMTUNE_TRUST_REMOTE_CODE")
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


def get_tokenizer(
    model_name_or_path: str,
    tokenizer_name_or_path: Optional[str] = None,
    additional_special_tokens: Optional[List[str]] = None,
    additional_tokens: Optional[List[str]] = None,
    pad_token: Optional[str] = None,
    eos_token: Optional[str] = None,
    bos_token: Optional[str] = None,
) -> Tuple[PreTrainedTokenizer, int]:
    """
    Get and optionally customize the tokenizer for the model.

    Returns:
        A tuple of (tokenizer, num_new_tokens) where num_new_tokens is the
        count of tokens added via additional_special_tokens and additional_tokens.
    """
    source = tokenizer_name_or_path if tokenizer_name_or_path else model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=source,
        padding_side="left",
        use_fast=False,
        trust_remote_code=resolve_trust_remote_code(),
    )

    num_new_tokens = 0

    if additional_special_tokens:
        existing = tokenizer.additional_special_tokens or []
        new_tokens = [t for t in additional_special_tokens if t not in existing]
        if new_tokens:
            num_new_tokens += tokenizer.add_special_tokens({"additional_special_tokens": existing + new_tokens})

    if additional_tokens:
        num_new_tokens += tokenizer.add_tokens(additional_tokens)

    if eos_token is not None:
        tokenizer.eos_token = eos_token
    if bos_token is not None:
        tokenizer.bos_token = bos_token

    if pad_token is not None:
        tokenizer.pad_token = pad_token
    else:
        # Preserve legacy behavior: if no explicit override, default pad to eos
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.model_max_length > 200_000:
        tokenizer.model_max_length = 16384

    if num_new_tokens > 0:
        logger.info(f"[AutoTune] Added {num_new_tokens} new token(s) to the tokenizer (vocab size: {len(tokenizer)})")

    return tokenizer, num_new_tokens


def resize_model_embeddings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_new_tokens: int = 0,
    pad_to_multiple_of: int = 64,
) -> None:
    """
    Resize model token embeddings when the tokenizer has been extended.
    No-op when num_new_tokens == 0 (preserves existing behavior).
    """
    if num_new_tokens <= 0:
        return

    current_size = model.get_input_embeddings().weight.shape[0]
    target_size = len(tokenizer)

    if pad_to_multiple_of > 1:
        target_size = (target_size + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

    if target_size == current_size:
        return

    model.resize_token_embeddings(target_size)

    input_embeddings = model.get_input_embeddings().weight.data
    avg_embedding = input_embeddings[:-num_new_tokens].mean(dim=0)
    input_embeddings[-num_new_tokens:] = avg_embedding

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_data = output_embeddings.weight.data
        avg_output = output_data[:-num_new_tokens].mean(dim=0)
        output_data[-num_new_tokens:] = avg_output

    logger.info(f"[AutoTune] Resized embeddings: {current_size} -> {target_size} ({num_new_tokens} new tokens)")


def extract_tokenizer_kwargs(training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tokenizer customization keys from a training_config dict."""
    keys = [
        "tokenizer_name_or_path",
        "additional_special_tokens",
        "additional_tokens",
        "pad_token",
        "eos_token",
        "bos_token",
    ]
    return {k: training_config[k] for k in keys if training_config.get(k) is not None}


def tokenize_batch(
    batch: Union[pd.DataFrame, Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    input_col: str = "input",
    output_col: str = "output",
    max_length: int = 1024,
    stride: int = 512,
) -> Dict[str, List[List[int]]]:
    """
    Tokenizes a batch of input-output pairs for LLM fine-tuning.
    Loss is computed only on output tokens.

    Args:
        batch: Pandas DataFrame (Ray) or list of dicts (HF).
        input_col: Column name for input text.
        output_col: Column name for output text.
        max_length: Maximum sequence length.
        stride: Sliding window stride for long inputs.

    Returns:
        Dict with keys: input_ids, attention_mask, labels
    """
    # Normalize batch to list of dicts
    if isinstance(batch, pd.DataFrame):
        examples = batch.to_dict(orient="records")
    else:
        examples = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]

    input_ids_list, attention_masks_list, labels_list = [], [], []

    # Guard against negative or 0 stride
    if stride <= 0:
        raise ValueError("stride must be > 0")

    # Ensure stride is smaller than max_length
    if stride > max_length:
        stride = max_length // 2

    for ex in examples:
        input_text = ex[input_col]
        output_text = ex[output_col]

        # Tokenize output separately
        output_tokens = tokenizer(str(output_text), add_special_tokens=False)["input_ids"]

        # Check if enough capacity
        capacity = max_length - len(output_tokens) - 1
        if capacity < 0:
            raise ValueError(f"Not enough capacity for output + EOS at given max_length = {max_length} tokens.")

        # Sliding window over input
        input_tokens = tokenizer(str(input_text), add_special_tokens=False)["input_ids"]
        start = 0
        while start < len(input_tokens):
            end = start + (max_length - len(output_tokens) - 1)  # Reserve space for output + EOS
            chunk = input_tokens[start:end]
            combined = chunk + output_tokens + [tokenizer.eos_token_id]

            # Attention mask
            attention_mask = [1] * len(combined)

            # Labels: mask input tokens with -100
            labels = [-100] * len(chunk) + output_tokens + [tokenizer.eos_token_id]

            # Pad if needed
            pad_len = max_length - len(combined)
            if pad_len > 0:  # pad left
                combined = [tokenizer.pad_token_id] * pad_len + combined
                attention_mask = [0] * pad_len + attention_mask
                labels = [-100] * pad_len + labels

            input_ids_list.append(combined)
            attention_masks_list.append(attention_mask)
            labels_list.append(labels)

            start += stride  # Move sliding window

    return {"input_ids": input_ids_list, "attention_mask": attention_masks_list, "labels": labels_list}


# Set the random seed globally
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def get_best_result(result_grid: ResultGrid, metric: str, mode: str) -> Result:
    """
    Get the best result from the result grid (tune).

    Args:
        result_grid: ResultGrid
            The data structure containing the results of ray.tune.
        metric: str
            The metric that is tracked.
        mode: str
            The optimization mode, either `min` or `max`.

    Returns:
        An instance of Result containing the details of the best performing
        trial according to the input `metric` and optimization `mode`.
    """
    assert mode in ["min", "max"]

    best_result = None
    best_metric = np.inf if mode == "min" else -np.inf
    for result in result_grid:
        if result.error:
            continue  # skip errors
        if mode == "min":
            if result.metrics[metric] < best_metric:
                best_metric = result.metrics[metric]
                best_result = result
        else:
            if result.metrics[metric] > best_metric:
                best_metric = result.metrics[metric]
                best_result = result

    return best_result


def cleanup(output_dir: str):
    """
    Remove the content from the output dir.

    Args:
        output_dir: str
            The output dir to remove content from.
    """

    output_path = Path(output_dir)
    folders = [
        output_path / "ray_results",
        output_path / "train_results",
        output_path / "lightning_logs",
        # Worker bring-up logs: <output_dir>/logs/ray_nodes_<ts>/<host>.log,
        # written by ray_up_blaunch. Owned by the launcher; wipe
        # along with the other transient artifacts.
        output_path / "logs",
    ]

    for folder in folders:
        if not folder.exists():
            continue

        print(f"Removing content from dir: {folder}")

        try:
            # Remove entire folder and recreate an empty one so downstream
            # code can rely on the transient dirs existing.
            shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"[AutoTune] Successfully cleaned: {folder}")
        except Exception as e:
            logger.info(f"[AutoTune] Failed to clean {folder}. Reason: {e}")


def remove_dir(folder: str):
    """
    Remove the content from the output dir.

    Args:
        output_dir: str
            The output dir to remove content from.
    """

    # Check if folder exists
    if not os.path.exists(folder):
        return

    # Delete all the files in the folder
    print(f"Removing content from dir: {folder}")
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            os.rmdir(folder)
        except Exception as e:
            logger.info("Failed to delete %s. Reason: %s" % (file_path, e))


def move_dir(source_dir: str, target_dir: str):
    try:
        # assumes that target dir already exists
        files = os.listdir(source_dir)
        for f in files:
            shutil.move(os.path.join(source_dir, f), os.path.join(target_dir, f))
    except IOError:
        # do nothing
        pass


def get_kwargs(config: dict, keyword: str, excluded: list):
    kwargs = {}
    for key, val in config.items():
        if key.startswith(keyword) and key not in excluded:
            k = key.split(".")[-1]
            kwargs[k] = val

    return kwargs


def get_adapter_compatible_modules(model, alora: bool = False) -> Union[str, List[str]]:
    """
    Get the names of the modules that are compatible with LoRA or aLoRA.
    For aLoRA, we only keep the q_proj, k_proj, v_proj modules. Otherwise,
    we return all linear layers in the model.
    """
    return "all-linear"
    # return "all-linear" if not alora else ["q_proj", "k_proj", "v_proj", "o_proj"]


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, ValueError):
        return False


def _json_safe(obj):
    """Return a structurally-identical copy of ``obj`` that is JSON-serializable.

    Dict keys are preserved (never dropped); list/tuple structure is preserved
    (tuples become lists). Any leaf value that is not JSON-serializable is
    replaced by its ``str()`` form. Already-serializable inputs are returned
    unchanged (fast path), so clean configs are byte-identical after a round-trip.

    Used by :func:`save_final_config` so a config that picked up non-serializable
    objects (e.g. Ray search-alg / scheduler instances inside ``tune_config``)
    can still be saved. Those objects are not consumed on resume — they are
    discarded and rebuilt — so stringifying them is lossless for resume.
    """
    if is_jsonable(obj):
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return str(obj)


# --- Final-training config persistence (for --resume_from_checkpoint) ---------
#
# The resolved best/default config used for the final training round is saved to
# <output_dir>/final_checkpoints/final_config.json so that a later
# --resume_from_checkpoint run can read it back, skip HPO entirely, and resume
# the final training from the last checkpoint in the same directory. On a
# successful final run the whole final_checkpoints/ dir (and this file with it)
# is cleaned up; only an interrupted run leaves both the checkpoint and config
# behind to resume from.

FINAL_CHECKPOINTS_DIRNAME = "final_checkpoints"
FINAL_CONFIG_FILENAME = "final_config.json"


def final_checkpoints_dir(output_dir: str) -> str:
    """Return the stable final-training checkpoint dir for an output dir."""
    return os.path.join(output_dir, FINAL_CHECKPOINTS_DIRNAME)


def final_config_path(output_dir: str) -> str:
    """Return the path to the saved final-config JSON for an output dir."""
    return os.path.join(final_checkpoints_dir(output_dir), FINAL_CONFIG_FILENAME)


def save_final_config(output_dir: str, config: dict) -> Optional[str]:
    """Persist the resolved final/default config to ``final_config.json``.

    The config is sanitized via :func:`_json_safe` first, so non-serializable
    leaves (e.g. Ray search-alg / scheduler objects that ride along inside
    ``tune_config``) are stringified rather than crashing the save. Those values
    are never consumed on resume — they are rebuilt — so this is lossless for the
    resume path. Saving is a best-effort convenience: any failure is logged and
    swallowed (returns ``None``) so it can never abort the final training run.

    Returns the written path on success, or ``None`` on failure.
    """
    path = final_config_path(output_dir)
    try:
        safe = _json_safe(config)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fp:
            json.dump(safe, fp, indent=2)
        return path
    except Exception as e:
        logger.warning(f"[AutoTune] Failed to save final config to {path!r}: {e}")
        return None


def load_final_config(output_dir: str) -> dict:
    """Load the saved final-config dict. Raises if the file is absent."""
    with open(final_config_path(output_dir)) as fp:
        return json.load(fp)


def has_resumable_final_checkpoint(output_dir: str) -> bool:
    """True iff ``final_checkpoints/`` holds a saved config AND a checkpoint.

    Requires both ``final_config.json`` and at least one ``checkpoint-<step>``
    subdirectory (numeric suffix, a real directory). The checkpoint-detection
    rule mirrors the drivers' ``_resolve_resume_checkpoint`` so the head-node
    decision (skip HPO?) never disagrees with the worker-side resume. Pure /
    filesystem-only — unit-testable without Ray or Torch.
    """
    ckpt_dir = final_checkpoints_dir(output_dir)
    if not os.path.isfile(final_config_path(output_dir)):
        return False
    if not os.path.isdir(ckpt_dir):
        return False
    for name in os.listdir(ckpt_dir):
        if not name.startswith("checkpoint-"):
            continue
        if not os.path.isdir(os.path.join(ckpt_dir, name)):
            continue
        if name[len("checkpoint-") :].isdigit():
            return True
    return False


def get_autotune_precision(precision: str):
    """
    Get the AutotunePrecision for a given string.

    Args:
        precision: str
            The precision string (i.e., fp32, bf16, int8, int4).
    """
    if precision == "fp32":
        return autotune.constants.AutotunePrecision.FP32
    elif precision == "fp16":
        return autotune.constants.AutotunePrecision.BF16
    elif precision == "bf16":
        return autotune.constants.AutotunePrecision.BF16
    elif precision == "int8":
        return autotune.constants.AutotunePrecision.INT8
    elif precision == "int4":
        return autotune.constants.AutotunePrecision.INT4
    else:
        raise ValueError(f"Uknown precision level: {precision}")


def get_autotune_tuning_types() -> Dict[str, Any]:
    """
    Return the tuning types supported by AutoTune. Each tuning type has a
    `key` and its corresponding value is a dict containing a description and
    a PEFT tuning type.
    """

    return autotune.constants.AutotuneTuningTypes


def get_autotune_metrics() -> Dict[str, Any]:
    """
    Return the metrics supported by AutoTune. Each metric has a `key` and its
    corresponding value is a dict containing a description and additional info.
    """

    return autotune.constants.AutotuneMetrics


def save_hpo_history(
    result_grid: ResultGrid,
    output_dir: str,
    run_name: str,
    metric: str = "loss",
    mode: str = "min",
):
    """
    Save the HPO trials history.

    Args:
        result_grid: ResultGrid
            The results grid produced by a ray.tune run.
        metric: str
            The metric being optimized during HPO search.
        mode: str
            The direction of the optimization (min, max).
        output_dir: str
            The path to the output location.
        run_name: str
            The name of the exeriment (used to name the output files).
    """

    best_result = result_grid.get_best_result(metric=metric, mode=mode)
    best_config = best_result.metrics.pop("config")
    score_tune = best_result.metrics

    # Filter out not jsonable elements from the best config
    save_config = {k: v for k, v in best_config.items() if not isinstance(v, dict)}

    # Printing out the results
    print(f"[AutoTune] Best metrics: {best_result.metrics}")
    print(f"[AutoTune] Best config: {best_config}")
    print(f"[AutoTune] Result records found: {len(result_grid)}")

    # Save the history of the trials
    trials_data = []
    for i in range(len(result_grid)):
        result = result_grid[i]
        trials_data.append(flatten_dict(result.metrics))
    trials_df = pd.DataFrame.from_dict(trials_data)

    # Create the output dir on the cluster
    print("[AutoTune] Writing results to files...")
    try:
        results_dir = os.path.join(output_dir, "results")
        os.makedirs(results_dir, exist_ok=True)
        file_name_tune = os.path.join(results_dir, run_name + "_tune.json")
        tune_results = {**score_tune, **save_config}
        with open(file_name_tune, "w") as fp:
            json.dump(tune_results, fp)
        file_name_trials = os.path.join(results_dir, run_name + "_trials.csv")
        trials_df.to_csv(file_name_trials)
        print("[AutoTune] Finished writing the results.")
    except Exception as e:
        print(f"Error while writing the output files: {e}.")


def compute_ray_resources(gpus_per_trial: int, model_size: int) -> Dict[str, Any]:
    """
    Determine the number of concurent trials, memory requirements, etc.

    Args:
        gpus_per_trial: int
            The number of GPUs per trial (default is 1)
        model_size: int
            Model size in billion of params (e.g., 2b, 8b, etc.)

    Returns:
        A dict containing the resources required to run AutoTune.

        This method creates extra CPU RAM requirements on the machines.
        A rule of thumb for this implementation is that it needs
        O(18M/N*K) CPU RAM where M is the model size, N is the number shards,
        and K is the number of GPUs on a single machine.
    """

    result = {
        "max_concurrent_trials": 1,
        "num_workers_per_trial": 1,
        "memory_per_worker": 10 * 1024**3,  # 10 GB
        "cpus_per_worker": 4,
        "gpus_per_worker": 1,
    }

    # Get the ray cluster resources
    resources = ray.cluster_resources()
    num_gpus = resources.get("GPU", None)
    accelerator_type = None
    for k in resources.keys():
        if "accelerator_type" in k:
            accelerator_type = k.split(":")[1]
            break

    try:
        # Determine how many concurrent trials can be run
        max_concurrent_trials = int(num_gpus / gpus_per_trial)

        # Update the resources
        result["max_concurrent_trials"] = max(1, max_concurrent_trials)
        result["num_cpus_per_worker"] = 4 * gpus_per_trial
        result["num_workers_per_trial"] = gpus_per_trial
        result["memory_per_worker"] = int(18 * model_size / gpus_per_trial) * 1024**3
        if accelerator_type is not None:
            result["accelerator_type"] = accelerator_type
        return result
    except Exception:
        return None


def get_peft_config(
    model: PreTrainedModel,
    model_name_or_path: str,
    peft_type: PeftType,
    base_kwargs: Dict[str, Any],
    tokenizer=None,
):
    """
    Create the PeftConfig object.

    Args:
        peft_type: PeftType
            The type of peft-based finetuning.
        model_name_or_path: str
            The model name in HF or path to it.
        model: PreTrainedModel
            The pretrained model object.
        base_kwargs: Dict[str, Any]
            A dict containing the hyperparams.
        tokenizer: Optional[PreTrainedTokenizer]
            The tokenizer, required for aLoRA to tokenize the invocation_string.

    Returns:
        A PeftConfig object.
    """

    peft_config = None
    if peft_type == PeftType.LORA:
        print(f"[AutoTune] Current LORA configuration: {base_kwargs}")
        logger.info(f"[AutoTune] Current LORA configuration: {base_kwargs}")
        target_modules = get_adapter_compatible_modules(model)
        print(f"[AutoTune] LORA compatible modules: {target_modules}")
        logger.info(f"[AutoTune] LORA compatible modules: {target_modules}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
            r=base_kwargs.get("r"),
            lora_alpha=base_kwargs.get("lora_alpha"),
            lora_dropout=base_kwargs.get("lora_dropout"),
            bias=base_kwargs.get("bias"),
        )
    elif peft_type == PeftType.LOHA:
        print(f"[AutoTune] Current configuration: {base_kwargs}")
        logger.info(f"[AutoTune] Current LOHA configuration: {base_kwargs}")
        target_modules = get_adapter_compatible_modules(model)
        print(f"[AutoTune] LOHA compatible modules: {target_modules}")
        logger.info(f"[AutoTune] LOHA compatible modules: {target_modules}")

        peft_config = LoHaConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
            r=base_kwargs.get("r"),
            alpha=base_kwargs.get("lora_alpha"),
            rank_dropout=base_kwargs.get("rank_dropout"),
            module_dropout=base_kwargs.get("module_dropout"),
        )
    elif peft_type == PeftType.LOKR:
        print(f"[AutoTune] Current LOKR configuration: {base_kwargs}")
        logger.info(f"[AutoTune] Current LOKR configuration: {base_kwargs}")
        target_modules = get_adapter_compatible_modules(model)
        print(f"[AutoTune] LOKR compatible modules: {target_modules}")
        logger.info(f"[AutoTune] LOKR compatible modules: {target_modules}")

        peft_config = LoKrConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
            r=base_kwargs.get("r"),
            alpha=base_kwargs.get("lora_alpha"),
            rank_dropout=base_kwargs.get("rank_dropout"),
            module_dropout=base_kwargs.get("module_dropout"),
            decompose_both=base_kwargs.get("decompose_both"),
        )
    elif peft_type == PeftType.VERA:
        print(f"[AutoTune] Current VERA configuration: {base_kwargs}")
        logger.info(f"[AutoTune] Current VERA configuration: {base_kwargs}")
        target_modules = get_adapter_compatible_modules(model)
        print(f"[AutoTune] VERA compatible modules: {target_modules}")
        logger.info(f"[AutoTune] VERA compatible modules: {target_modules}")

        peft_config = VeraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,
            r=base_kwargs.get("r"),
            vera_dropout=base_kwargs.get("vera_dropout"),
            bias=base_kwargs.get("bias"),
            d_initial=base_kwargs.get("d_initial"),
        )
    elif peft_type == "ALORA":
        # aLoRA (PEFT 0.18 native): uses LoraConfig with alora_invocation_tokens.
        # The adapter activates only after the invocation token sequence appears
        # in the input, preserving base-model KV cache before that point.
        if tokenizer is None:
            raise ValueError("aLoRA requires a tokenizer to encode the invocation_string into alora_invocation_tokens.")

        # Install the peft 0.18 GC-compatibility patch so gradient checkpointing
        # can be used with aLoRA. Drivers gate `gradient_checkpointing` on
        # `alora_patch.is_active()`; if the patch no-ops (wrong peft version),
        # they fall back to disabling GC for aLoRA.
        from autotune.alora_patch import apply_alora_gc_patch

        apply_alora_gc_patch()

        print(f"[AutoTune] Current aLORA configuration: {base_kwargs}")
        logger.info(f"[AutoTune] Current aLORA configuration: {base_kwargs}")
        target_modules = get_adapter_compatible_modules(model, alora=True)
        print(f"[AutoTune] aLORA compatible modules: {target_modules}")
        logger.info(f"[AutoTune] aLORA compatible modules: {target_modules}")

        invocation_string = base_kwargs.get("invocation_string", "<guardian>")
        inv_tokens = tokenizer.encode(invocation_string, add_special_tokens=False)
        print(f"[AutoTune] aLoRA invocation_string='{invocation_string}' → tokens={inv_tokens}")
        logger.info(f"[AutoTune] aLoRA invocation_tokens={inv_tokens}")

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=target_modules,  # aLoRA must only adapt q, k, v layers.
            r=base_kwargs.get("r"),
            lora_alpha=base_kwargs.get("lora_alpha"),
            lora_dropout=base_kwargs.get("lora_dropout"),
            bias=base_kwargs.get("bias"),
            alora_invocation_tokens=inv_tokens,
        )
    else:
        peft_config = None  # SFT

    return peft_config


def get_qlora_quantization_config(compute_dtype=torch.bfloat16):
    """
    Build the 4-bit (NF4) BitsAndBytesConfig used for QLoRA.

    QLoRA quantizes the frozen base model to 4-bit NF4 (with double quantization)
    while the LoRA adapter is trained in ``compute_dtype``. This config is passed
    as ``quantization_config`` to ``AutoModelForCausalLM.from_pretrained`` by the
    drivers when the tuning algorithm is ``qlora``.

    Args:
        compute_dtype: torch.dtype
            The dtype used for the forward/backward compute (default bf16).

    Returns:
        A ``BitsAndBytesConfig`` configured for 4-bit NF4 double quantization.
    """

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def prepare_qlora_model(model, use_gradient_checkpointing=True):
    """
    Prepare a 4-bit quantized model for QLoRA (k-bit) training.

    Wraps ``peft.prepare_model_for_kbit_training`` — casts layernorms to fp32,
    enables input-require-grads, and (optionally) sets up gradient checkpointing.
    Must be called on the quantized base before ``get_peft_model``.
    """

    return prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_gradient_checkpointing)


def _extract_sampler(dataloader):
    """Return the most-specific sampler attached to ``dataloader``.

    After ``accelerator.prepare()`` wraps a DataLoader for distributed
    training, the DistributedSampler typically sits under
    ``dataloader.batch_sampler.sampler``, not ``dataloader.sampler``.
    Walk both paths and prefer the inner one when present.
    """
    if dataloader is None:
        return None
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    inner = getattr(batch_sampler, "sampler", None)
    if inner is not None:
        return inner
    return getattr(dataloader, "sampler", None)


def _describe_batch_sampler(dataloader):
    """Describe the batch_sampler and whether it implements DP sharding itself.

    Accelerate (≥1.x) does DP sharding by wrapping the DataLoader's
    batch_sampler in ``BatchSamplerShard(num_processes=..., process_index=...)``
    rather than replacing the per-sample sampler with DistributedSampler.
    In that case ``dataloader.sampler`` stays whatever HF Trainer created
    (RandomSampler / SeedableRandomSampler), and the sharding lives on the
    batch_sampler. Return (class_name, num_processes, process_index) so the
    audit can recognize that path as correct.
    """
    if dataloader is None:
        return ("None", None, None)
    bs = getattr(dataloader, "batch_sampler", None)
    if bs is None:
        return ("None", None, None)
    return (
        type(bs).__name__,
        getattr(bs, "num_processes", None),
        getattr(bs, "process_index", None),
    )


def assert_dp_sharding(
    trainer,
    rank: int,
    world_size: int,
    per_device_batch_size: int,
    gradient_accumulation_steps: int,
    steps_per_epoch: int,
    num_train_epochs: int,
) -> None:
    """
    Verify data-parallel sharding is wired correctly and log global batch math.
    Logs (rank 0 only):
      - Train/eval sampler class, num_replicas, rank — should be
        DistributedSampler(num_replicas=world_size) once Accelerate has
        wrapped the dataloader.
      - Global effective batch size = per_device * world_size * grad_accum.
      - HF Trainer's computed max_steps vs the driver's
        steps_per_epoch * num_train_epochs. Warns if they disagree.

    Call this **after** ``trainer.train()`` has returned. Calling it before
    training starts sees the pre-Accelerate-prepare dataloader, whose sampler
    is typically RandomSampler/SequentialSampler — the DistributedSampler is
    installed by ``accelerator.prepare_data_loader()`` inside ``train()``.
    """
    if rank != 0:
        return

    try:
        train_loader = trainer.get_train_dataloader()
        train_sampler = _extract_sampler(train_loader)
        sampler_cls = type(train_sampler).__name__ if train_sampler is not None else "None"
        num_replicas = getattr(train_sampler, "num_replicas", None)
        sampler_rank = getattr(train_sampler, "rank", None)
        bs_cls, bs_nprocs, bs_procidx = _describe_batch_sampler(train_loader)

        eval_loader = None
        try:
            eval_loader = trainer.get_eval_dataloader()
        except Exception:
            pass
        eval_sampler = _extract_sampler(eval_loader)
        eval_sampler_cls = type(eval_sampler).__name__ if eval_sampler is not None else "None"

        effective_batch = per_device_batch_size * world_size * max(1, gradient_accumulation_steps)
        expected_max_steps = max(1, steps_per_epoch * max(1, num_train_epochs))
        hf_max_steps = getattr(trainer.state, "max_steps", None) or getattr(trainer.args, "max_steps", None)

        logger.info(
            "[DP audit] train_sampler=%s(num_replicas=%s, rank=%s) "
            "train_batch_sampler=%s(num_processes=%s, process_index=%s) "
            "eval_sampler=%s world_size=%s per_device_bs=%s grad_accum=%s "
            "effective_bs=%s steps_per_epoch=%s epochs=%s "
            "expected_max_steps=%s hf_max_steps=%s",
            sampler_cls,
            num_replicas,
            sampler_rank,
            bs_cls,
            bs_nprocs,
            bs_procidx,
            eval_sampler_cls,
            world_size,
            per_device_batch_size,
            gradient_accumulation_steps,
            effective_batch,
            steps_per_epoch,
            num_train_epochs,
            expected_max_steps,
            hf_max_steps,
        )

        # Correctness — sharding is OK if ANY of:
        #  1. The per-sample sampler is DistributedSampler (older Accelerate /
        #     direct HF path). num_replicas must match world_size.
        #  2. The batch_sampler is BatchSamplerShard (newer Accelerate ≥1.x —
        #     sharding is done by filtering batches, not indices). The
        #     per-sample sampler stays RandomSampler/SeedableRandomSampler;
        #     this is EXPECTED, not a problem.
        #  3. The per-sample sampler is _InfiniteConstantSampler (iterable
        #     dataset; sharding is upstream, e.g., Ray Data SplitCoordinator).
        _ITERABLE_SAMPLER_SENTINELS = {"_InfiniteConstantSampler"}
        _BATCH_SAMPLER_SHARD_NAMES = {"BatchSamplerShard"}

        sharded_by_sampler = sampler_cls == "DistributedSampler"
        sharded_by_batch = bs_cls in _BATCH_SAMPLER_SHARD_NAMES and bs_nprocs is not None and bs_nprocs == world_size
        sharded_by_iterable = sampler_cls in _ITERABLE_SAMPLER_SENTINELS

        if world_size > 1:
            if sharded_by_sampler:
                logger.info(
                    "[DP audit] DP sharding OK — DistributedSampler (num_replicas=%s, rank=%s).",
                    num_replicas,
                    sampler_rank,
                )
            elif sharded_by_batch:
                logger.info(
                    "[DP audit] DP sharding OK — Accelerate %s(num_processes=%s, "
                    "process_index=%s) on batch_sampler; per-sample sampler %s "
                    "is same-on-every-rank by design.",
                    bs_cls,
                    bs_nprocs,
                    bs_procidx,
                    sampler_cls,
                )
            elif sharded_by_iterable:
                logger.info(
                    "[DP audit] DP sharding delegated upstream — iterable "
                    "dataset (sampler=%s), e.g., Ray Data SplitCoordinator.",
                    sampler_cls,
                )
            else:
                logger.warning(
                    "[DP audit] world_size=%s but no recognized sharding "
                    "mechanism: sampler=%s (expected DistributedSampler) and "
                    "batch_sampler=%s (expected BatchSamplerShard). Every rank "
                    "may be training on the full dataset.",
                    world_size,
                    sampler_cls,
                    bs_cls,
                )

        if num_replicas is not None and num_replicas != world_size:
            logger.warning(
                "[DP audit] DistributedSampler num_replicas=%s != world_size=%s",
                num_replicas,
                world_size,
            )
        if bs_nprocs is not None and bs_nprocs != world_size:
            logger.warning(
                "[DP audit] BatchSamplerShard num_processes=%s != world_size=%s",
                bs_nprocs,
                world_size,
            )
        if hf_max_steps is not None and hf_max_steps != expected_max_steps:
            logger.warning(
                "[DP audit] HF max_steps=%s disagrees with driver "
                "expected_max_steps=%s (steps_per_epoch=%s * epochs=%s). "
                "Check gradient_accumulation_steps and dataset length.",
                hf_max_steps,
                expected_max_steps,
                steps_per_epoch,
                num_train_epochs,
            )
    except Exception as exc:
        logger.warning("[DP audit] skipped due to error: %s", exc)


def find_checkpoints(root_dir: str):
    checkpoint_paths = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # print(f"Searching for checkpoints in: {dirpath}")
        # print(f"Found directories: {dirnames}")
        for dirname in dirnames:
            if dirname in ["checkpoint.ckpt", "model.pt"]:
                ckpt_path = os.path.join(dirpath, dirname)
                checkpoint_paths.append(ckpt_path)

    return checkpoint_paths


def find_latest_checkpoint(base_dir: str, ckpt_type: str = "hf") -> Optional[str]:
    """
    Find the latest HuggingFace checkpoint in the specified base directory.

    Args:
        base_dir: str
            The base directory where the training runs are stored.

    """

    base_dir = os.path.expanduser(base_dir)
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        return None

    # Find the most recent checkpoint directory (i.e., checkpoint-123)
    if ckpt_type == "hf":
        checkpoints = [
            os.path.join(base_dir, d)
            for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("checkpoint-")
        ]
    elif ckpt_type == "pl":
        checkpoints = find_checkpoints(base_dir)
    else:
        print(f"Unsupported checkpoint type: {ckpt_type}")
        checkpoints = []

    if not checkpoints:
        print(f"No checkpoints found in {base_dir}.")
        return None

    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt


def generate_unique_id(length=8):
    """
    Generate a unique ID of size 8 containing letters and digits
    """

    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    unique_id = "".join(random.choices(characters, k=length))
    return unique_id


def resolve_model_path(model_name_or_path: str) -> str:
    """
    Resolve a model path that may point to a parent directory of the actual
    model files. If the given path is a local directory that does not contain
    config.json at its root, search immediate subdirectories (one level deep)
    for config.json.

    HuggingFace model identifiers (e.g. 'facebook/opt-125m') are returned
    unchanged.

    Args:
        model_name_or_path: A HuggingFace model identifier or a local
            filesystem path to a model directory.

    Returns:
        The resolved model path string.

    Raises:
        ValueError: If config.json is found in multiple immediate
            subdirectories (ambiguous model location).
    """
    if not (
        model_name_or_path.startswith("/")
        or model_name_or_path.startswith("./")
        or model_name_or_path.startswith("../")
        or model_name_or_path.startswith("~")
    ):
        return model_name_or_path

    resolved = os.path.abspath(os.path.expanduser(model_name_or_path))

    if not os.path.isdir(resolved):
        logger.warning(f"[AutoTune] Model path '{resolved}' does not exist or is not a directory; returning as-is.")
        return resolved

    if os.path.isfile(os.path.join(resolved, "config.json")):
        return resolved

    candidates = []
    try:
        for entry in os.listdir(resolved):
            subdir = os.path.join(resolved, entry)
            if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "config.json")):
                candidates.append(subdir)
    except PermissionError as e:
        logger.warning(f"[AutoTune] Permission error scanning '{resolved}': {e}; returning original path.")
        return resolved

    if len(candidates) == 1:
        print(f"[AutoTune] Resolved model path: '{resolved}' -> '{candidates[0]}'")
        logger.info(f"[AutoTune] Resolved model path: '{resolved}' -> '{candidates[0]}'")
        return candidates[0]
    elif len(candidates) > 1:
        subdirs = [os.path.basename(c) for c in candidates]
        raise ValueError(
            f"[AutoTune] Ambiguous model path '{resolved}': config.json found "
            f"in multiple subdirectories: {subdirs}. Please specify the exact "
            f"model directory."
        )
    else:
        logger.warning(
            f"[AutoTune] No config.json found in '{resolved}' or its immediate subdirectories; returning original path."
        )
        return resolved


def get_model_parameters(model_name_or_path: str) -> Tuple[int, int]:
    """
    Get the total number of parameters in the model.

    Args:
        model_name_or_path: str
            The name of the model in HuggingFace or the path to the model directory.

    Returns:
        A tuple containing the total number of parameters and the number of parameters in billions.
        For example, if the model has 2 billion parameters, it will return (2_000_000_000, 2).
    """

    try:
        # Try loading as a causal language model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    except Exception:
        try:
            # Try loading as a sequence classification model
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        except Exception:
            # Fallback to generic AutoModel
            model = AutoModel.from_pretrained(model_name_or_path)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    billion_params = int(total_params / 1e9)

    print(f"[AutoTune] Total parameters in the model {model_name_or_path}: {total_params:,}")
    print(f"[AutoTune] Total parameters in the model {model_name_or_path}: {billion_params}B")

    return total_params, billion_params


def get_model_params_count(model_name_or_path: str) -> Tuple[int, int]:
    """
    Get the total number of parameters in the model.

    Args:
        model_name_or_path: str
            The name of the model in HuggingFace or the path to the model directory.

    Returns:
        A tuple containing the total number of parameters and the number of parameters in billions.
        For example, if the model has 2 billion parameters, it will return (2_000_000_000, 2).
    """

    config = AutoConfig.from_pretrained(model_name_or_path)

    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "n_layer", getattr(config, "num_hidden_layers", None))
    vocab_size = getattr(config, "vocab_size", None)
    intermediate_size = getattr(config, "n_inner", getattr(config, "intermediate_size", None))
    num_heads = getattr(config, "n_head", getattr(config, "num_attention_heads", None))

    if not all([hidden_size, num_layers, vocab_size, intermediate_size, num_heads]):
        raise ValueError("Config missing required attributes for estimation.")

    # Embedding parameters: token + positional
    embedding_params = vocab_size * hidden_size
    if hasattr(config, "n_positions"):
        embedding_params += config.n_positions * hidden_size

    # Attention parameters per layer:
    # Q, K, V, and output projections: 4 * hidden_size * hidden_size
    attention_params = 4 * hidden_size * hidden_size

    # Feed-forward parameters per layer:
    ff_params = 2 * hidden_size * intermediate_size

    # Total per layer
    layer_params = attention_params + ff_params

    # Total for all layers
    total_params = embedding_params + (num_layers * layer_params)
    billion_params = int(total_params / 1e9)

    print(f"[AutoTune] Total parameters in the model {model_name_or_path}: {total_params:,}")
    print(f"[AutoTune] Total parameters in the model {model_name_or_path}: {billion_params}B")

    return total_params, billion_params


def parse_model_parameters(model_name_or_path: str) -> float | None:
    """
    Extracts the number of parameters from a Hugging Face model name.
    Supports billions (B/b) and millions (M/m).

    Args:
        model_name (str):
            The full model name, e.g., 'meta-llama/Llama-2-7b' or 'google/gemma-2m'.

    Returns:
        float | None: Number of parameters in billions, or None if not found.
    """

    # Look for patterns like '7b', '13B', '500m', '1.3M', etc.
    match = re.search(r"(\d+(?:\.\d+)?)([bBmM])", model_name_or_path)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        return value if unit == "b" else value / 1000  # Convert millions to billions
    return None


def estimate_memory_usage(
    model_size_billion_params: float,
    precision: str = "bf16",
    batch_size: int = 1,
    sequence_length: int = 128,
    use_gradient_checkpointing: bool = True,
    zero_stage: int = 3,
    use_lora: bool = False,
    gpu_size_gb: int = 75,
) -> Dict[str, Any]:
    """
    Estimate the memory usage for training a model based on its size, precision,
    batch size, sequence length, and other parameters.

    Args:
        model_size_billion_params: int
            The size of the model in billions of parameters (e.g., 2,000,000,000 for 2B).
        precision: str
            The precision of the model weights (e.g., 'fp32', 'fp16', 'bf16', 'int8', 'int4').
        batch_size: int
            The batch size for training.
        sequence_length: int
            The sequence length for training.
        use_gradient_checkpointing: bool
            Whether to use gradient checkpointing to reduce memory usage.
        zero_stage: int
            The ZeRO optimization stage (0, 1, 2, or 3).
        use_lora: bool
            Whether to use LoRA for training, which reduces the number of trainable parameters.
        gpu_size_gb: int
            The size of the GPU memory in GB (default is 80GB for A100 GPUs).

    Returns:
        A dictionary containing the estimated total memory usage in GB, the number of GPUs needed,
        the CPU memory usage, and the breakdown of memory usage for weights,
        optimizer states, gradients, and activations.
    """

    # Overhead
    overhead = 1.5

    # CPU memory
    cpu_memory = 0.0

    # Define bytes per parameter for different precisions
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}

    if precision not in precision_bytes:
        raise ValueError("Unsupported precision. Choose from 'fp32', 'fp16', 'bf16', 'int8' or 'int4'.")

    bytes_per_param = precision_bytes[precision]

    # Convert model size to number of parameters
    num_params = model_size_billion_params * 1e9

    # Adjust trainable parameters for LoRA (assume 10% of full model size)
    trainable_params = num_params * 0.1 if use_lora else num_params

    # Memory for model weights
    weights_memory = num_params * bytes_per_param * overhead

    # Memory for optimizer states (typically 2-3x model size)
    optimizer_memory = 2 * trainable_params * bytes_per_param * overhead

    # Memory for gradients (same size as weights)
    gradients_memory = trainable_params * bytes_per_param * overhead

    # Memory for activations (depends on batch size and sequence length)
    # Assume each token produces one activation per parameter (simplified)
    activations_memory = batch_size * sequence_length * bytes_per_param * model_size_billion_params * 1e9 / 1024

    print(f"[AutoTune] Model memory (weights): {weights_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Optimizer memory: {optimizer_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Gradients memory: {gradients_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Activations memory: {activations_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Trainable parameters: {trainable_params / 1e6:.2f} M")

    # Apply gradient checkpointing reduction
    if use_gradient_checkpointing:
        new_activations_memory = activations_memory * 0.5
        cpu_memory += activations_memory - new_activations_memory
        activations_memory = new_activations_memory

    # Apply ZeRO optimization reductions
    if zero_stage >= 1:
        new_optimizer_memory = optimizer_memory * 0.25  # Assume 75% reduction
        cpu_memory += optimizer_memory - new_optimizer_memory
        optimizer_memory = new_optimizer_memory
    if zero_stage >= 2:
        new_gradients_memory = gradients_memory * 0.25  # Assume 75% reduction
        cpu_memory += gradients_memory - new_gradients_memory
        gradients_memory = new_gradients_memory
    if zero_stage == 3:
        new_weights_memory = weights_memory * 0.25  # Assume 75% reduction
        cpu_memory += weights_memory - new_weights_memory
        weights_memory = new_weights_memory

    print(f"[AutoTune] Applying ZeRO Stage 3: weights memory reduced to {weights_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Applying ZeRO Stage 3: optimizer memory reduced to {optimizer_memory / (1024**3):.2f} GB")
    print(f"[AutoTune] Applying ZeRO Stage 3: gradients memory reduced to {gradients_memory / (1024**3):.2f} GB")

    # Total memory in bytes
    total_memory_bytes = weights_memory + optimizer_memory + gradients_memory + activations_memory

    # Convert to GB
    gpu_memory_gb = total_memory_bytes / (1024**3)
    cpu_memory_gb = cpu_memory / (1024**3)

    # Predict the number of GPUs needed based on GPU memory size
    num_gpus = int(gpu_memory_gb / gpu_size_gb) + 1  # Round up to nearest whole GPU

    result = {
        "model_size_billion_params": model_size_billion_params,
        "gpu_memory_gb": gpu_memory_gb,
        "num_gpus": num_gpus,
        "cpu_memory_gb": cpu_memory_gb,
        "weights_memory": weights_memory / (1024**3),  # Convert to GB
        "optimizer_memory": optimizer_memory / (1024**3),  # Convert to GB
        "gradients_memory": gradients_memory / (1024**3),  # Convert to GB
        "activations_memory": activations_memory / (1024**3),  # Convert to GB
    }

    return result


def _estimate_memory_components(
    model_name_or_path: str,
    max_seq_length: int,
    per_device_batch_size: int,
    peft_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Estimate per-component memory (bytes) for distributed training strategy selection.

    Returns dict with: weights_mem, optimizer_mem, gradients_mem, activations_mem,
    total_params, and trainable_params.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=resolve_trust_remote_code())

    hidden_size = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    vocab_size = getattr(config, "vocab_size", None)
    intermediate_size = getattr(config, "intermediate_size", getattr(config, "n_inner", None))

    if not all([hidden_size, num_layers, vocab_size, intermediate_size]):
        raise ValueError(
            f"Model config for {model_name_or_path} is missing required "
            f"attributes (hidden_size, num_hidden_layers, vocab_size, "
            f"intermediate_size)."
        )

    # Total model parameters (same formula as get_model_params_count)
    embedding_params = vocab_size * hidden_size
    attention_params = 4 * hidden_size * hidden_size
    ff_params = 2 * hidden_size * intermediate_size
    layer_params = attention_params + ff_params
    total_params = embedding_params + (num_layers * layer_params)

    # Trainable parameters depend on PEFT config
    if peft_config and peft_config.get("r"):
        r = peft_config["r"]
        # LoRA with target_modules="all-linear": ~7 linear layers per
        # transformer block (Q, K, V, O projections + gate, up, down FFN).
        # Each adapted module adds 2 * r * dim params (A and B matrices).
        # Attention modules have dim=hidden_size, FFN modules have
        # dim=intermediate_size (gate/up) or hidden_size (down).
        attn_lora = 4 * 2 * r * hidden_size  # Q, K, V, O
        ffn_lora = 2 * 2 * r * intermediate_size + 2 * r * hidden_size  # gate, up, down
        trainable_params = num_layers * (attn_lora + ffn_lora)
    else:
        trainable_params = total_params

    bytes_per_param = 2  # bf16

    weights_mem = total_params * bytes_per_param
    optimizer_mem = trainable_params * 8  # AdamW: 2 fp32 states = 8 bytes/param
    gradients_mem = trainable_params * bytes_per_param
    # Activations: reduced by gradient checkpointing (~sqrt(num_layers) factor)
    ckpt_factor = num_layers**0.5 / num_layers
    activations_mem = per_device_batch_size * max_seq_length * hidden_size * num_layers * bytes_per_param * ckpt_factor

    return {
        "weights_mem": weights_mem,
        "optimizer_mem": optimizer_mem,
        "gradients_mem": gradients_mem,
        "activations_mem": activations_mem,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def estimate_fsdp_strategy(
    model_name_or_path: str,
    max_seq_length: int,
    per_device_batch_size: int,
    num_gpus: int,
    peft_config: Optional[Dict[str, Any]] = None,
    gpu_memory_gb: int = 75,
) -> str:
    """
    Estimate the fastest FSDP sharding strategy that fits in GPU memory.

    Tries strategies from least sharding (fastest) to most sharding:
    no_shard (DDP) → shard_grad_op → full_shard.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        max_seq_length: Maximum sequence length for training.
        per_device_batch_size: Batch size per GPU.
        num_gpus: Number of A100 80GB GPUs allocated.
        peft_config: Optional dict with PEFT params (e.g. {"r": 128, "peft_type": "lora"}).
            If None, assumes full fine-tuning.
        gpu_memory_gb: Per-GPU memory in GB (default 80 for A100).

    Returns:
        One of "no_shard", "shard_grad_op", "full_shard".

    Raises:
        ValueError: If the model doesn't fit even with full_shard.
    """
    m = _estimate_memory_components(model_name_or_path, max_seq_length, per_device_batch_size, peft_config)
    weights_mem = m["weights_mem"]
    optimizer_mem = m["optimizer_mem"]
    gradients_mem = m["gradients_mem"]
    activations_mem = m["activations_mem"]
    total_params = m["total_params"]
    trainable_params = m["trainable_params"]

    GB = 1024**3
    overhead = 1.5  # CUDA allocator fragmentation, NCCL buffers, etc.
    gpu_capacity = gpu_memory_gb * GB

    # Print model summary
    print(f"[AutoTune] FSDP strategy estimation for {model_name_or_path}")
    print(f"[AutoTune]   Total params:     {total_params / 1e9:.2f}B")
    print(f"[AutoTune]   Trainable params: {trainable_params / 1e6:.1f}M")
    print(
        f"[AutoTune]   GPUs: {num_gpus}x {gpu_memory_gb}GB, "
        f"batch_size={per_device_batch_size}, seq_len={max_seq_length}"
    )
    print("[AutoTune]   Base memory (unsharded per GPU):")
    print(f"[AutoTune]     Weights:     {weights_mem / GB:.2f} GB")
    print(f"[AutoTune]     Optimizer:   {optimizer_mem / GB:.2f} GB")
    print(f"[AutoTune]     Gradients:   {gradients_mem / GB:.2f} GB")
    print(f"[AutoTune]     Activations: {activations_mem / GB:.2f} GB (with grad checkpointing)")
    print(f"[AutoTune]     Overhead:    {overhead}x")

    # Try strategies from fastest (least sharding) to slowest (most sharding)
    strategies = [
        (
            "no_shard",
            {
                "weights": weights_mem,
                "optimizer": optimizer_mem,
                "gradients": gradients_mem,
            },
        ),
        (
            "shard_grad_op",
            {
                "weights": weights_mem,
                "optimizer": optimizer_mem / num_gpus,
                "gradients": gradients_mem / num_gpus,
            },
        ),
        (
            "full_shard",
            {
                "weights": weights_mem / num_gpus,
                "optimizer": optimizer_mem / num_gpus,
                "gradients": gradients_mem / num_gpus,
            },
        ),
    ]

    for strategy, mem in strategies:
        total_mem = (mem["weights"] + mem["optimizer"] + mem["gradients"] + activations_mem) * overhead
        total_mem_gb = total_mem / GB
        fits = "FITS" if total_mem <= gpu_capacity else "OOM"

        print(
            f"[AutoTune]   {strategy:15s}: "
            f"weights={mem['weights'] / GB:.2f} "
            f"optim={mem['optimizer'] / GB:.2f} "
            f"grads={mem['gradients'] / GB:.2f} "
            f"act={activations_mem / GB:.2f} "
            f"→ {total_mem_gb:.1f} GB/GPU [{fits}]"
        )

        if total_mem <= gpu_capacity:
            print(f"[AutoTune] Selected FSDP strategy: {strategy}")
            return strategy

    raise ValueError(
        f"Model {model_name_or_path} ({total_params / 1e9:.1f}B params) "
        f"does not fit in {num_gpus}x {gpu_memory_gb}GB GPUs even with "
        f"full_shard. Reduce batch size or sequence length."
    )


def estimate_ds_strategy(
    model_name_or_path: str,
    max_seq_length: int,
    per_device_batch_size: int,
    num_gpus: int,
    peft_config: Optional[Dict[str, Any]] = None,
    gpu_memory_gb: int = 75,
) -> str:
    """
    Estimate the fastest DeepSpeed ZeRO strategy that fits in GPU memory.

    Tries strategies from fastest (least offloading) to most aggressive:
    zero1_gpu → zero2_gpu → zero3_gpu → zero2_cpu → zero3_cpu.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        max_seq_length: Maximum sequence length for training.
        per_device_batch_size: Batch size per GPU.
        num_gpus: Number of A100 80GB GPUs allocated.
        peft_config: Optional dict with PEFT params (e.g. {"r": 128, "peft_type": "lora"}).
            If None, assumes full fine-tuning.
        gpu_memory_gb: Per-GPU memory in GB (default 80 for A100).

    Returns:
        One of "zero1_gpu", "zero2_gpu", "zero3_gpu", "zero2_cpu", "zero3_cpu".

    Raises:
        ValueError: If the model doesn't fit even with zero3_cpu.
    """
    m = _estimate_memory_components(model_name_or_path, max_seq_length, per_device_batch_size, peft_config)
    weights_mem = m["weights_mem"]
    optimizer_mem = m["optimizer_mem"]
    gradients_mem = m["gradients_mem"]
    activations_mem = m["activations_mem"]
    total_params = m["total_params"]
    trainable_params = m["trainable_params"]

    GB = 1024**3
    overhead = 1.5  # CUDA allocator fragmentation, NCCL buffers, etc.
    gpu_capacity = gpu_memory_gb * GB

    # Print model summary
    print(f"[AutoTune] DeepSpeed strategy estimation for {model_name_or_path}")
    print(f"[AutoTune]   Total params:     {total_params / 1e9:.2f}B")
    print(f"[AutoTune]   Trainable params: {trainable_params / 1e6:.1f}M")
    print(
        f"[AutoTune]   GPUs: {num_gpus}x {gpu_memory_gb}GB, "
        f"batch_size={per_device_batch_size}, seq_len={max_seq_length}"
    )
    print("[AutoTune]   Base memory (unsharded per GPU):")
    print(f"[AutoTune]     Weights:     {weights_mem / GB:.2f} GB")
    print(f"[AutoTune]     Optimizer:   {optimizer_mem / GB:.2f} GB")
    print(f"[AutoTune]     Gradients:   {gradients_mem / GB:.2f} GB")
    print(f"[AutoTune]     Activations: {activations_mem / GB:.2f} GB (with grad checkpointing)")
    print(f"[AutoTune]     Overhead:    {overhead}x")

    # Try strategies from fastest to most aggressive offloading.
    # ZeRO1: shard optimizer states only
    # ZeRO2: shard optimizer states + gradients
    # ZeRO3: shard weights + optimizer states + gradients (like FSDP FULL_SHARD)
    # ZeRO2+CPU: offload optimizer to CPU, shard gradients
    # ZeRO3+CPU: shard weights + gradients, offload optimizer to CPU
    strategies = [
        (
            "zero1_gpu",
            {
                "weights": weights_mem,
                "optimizer": optimizer_mem / num_gpus,
                "gradients": gradients_mem,
            },
        ),
        (
            "zero2_gpu",
            {
                "weights": weights_mem,
                "optimizer": optimizer_mem / num_gpus,
                "gradients": gradients_mem / num_gpus,
            },
        ),
        (
            "zero3_gpu",
            {
                "weights": weights_mem / num_gpus,
                "optimizer": optimizer_mem / num_gpus,
                "gradients": gradients_mem / num_gpus,
            },
        ),
        (
            "zero2_cpu",
            {
                "weights": weights_mem,
                "optimizer": 0,  # offloaded to CPU
                "gradients": gradients_mem / num_gpus,
            },
        ),
        (
            "zero3_cpu",
            {
                "weights": weights_mem / num_gpus,
                "optimizer": 0,  # offloaded to CPU
                "gradients": gradients_mem / num_gpus,
            },
        ),
    ]

    for strategy, mem in strategies:
        total_mem = (mem["weights"] + mem["optimizer"] + mem["gradients"] + activations_mem) * overhead
        total_mem_gb = total_mem / GB
        fits = "FITS" if total_mem <= gpu_capacity else "OOM"

        optim_label = "cpu" if mem["optimizer"] == 0 else f"{mem['optimizer'] / GB:.2f}"
        print(
            f"[AutoTune]   {strategy:15s}: "
            f"weights={mem['weights'] / GB:.2f} "
            f"optim={optim_label:>5s} "
            f"grads={mem['gradients'] / GB:.2f} "
            f"act={activations_mem / GB:.2f} "
            f"→ {total_mem_gb:.1f} GB/GPU [{fits}]"
        )

        if total_mem <= gpu_capacity:
            print(f"[AutoTune] Selected DeepSpeed strategy: {strategy}")
            return strategy

    raise ValueError(
        f"Model {model_name_or_path} ({total_params / 1e9:.1f}B params) "
        f"does not fit in {num_gpus}x {gpu_memory_gb}GB GPUs even with "
        f"zero3_cpu. Reduce batch size or sequence length."
    )
