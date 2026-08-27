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
import argparse
import json
import logging
import os
import sys

import ray

from autotune.bridge_setup import resolve_bridge_settings
from autotune.callbacks.autotunex_api import AutoTuneXAPI, get_user_details
from autotune.callbacks.logging_service import BufferedLogHandler
from autotune.callbacks.print_logger import PrintLogger
from autotune.callbacks.tuner_callback import CustomLoggerCallback
from autotune.cluster import (
    configure_ray_data_context,
    start_local_ray_cluster,
    stop_local_ray_cluster,
)
from autotune.config import AutotuneConfig
from autotune.device import (
    apply_platform_guards,
    configure_runtime_env,
    detect_accelerator,
)
from autotune.logging_setup import setup_logging
from autotune.lsf import (
    RayUpTimeoutError,
    _rdma_env,
    start_multinode_ray_cluster_blaunch,
    stop_multinode_ray_cluster,
)

# Local
from autotune.optimizer import AutotuneOptimizer
from autotune.pipeline import AutotunePipeline
from autotune.template_utils import resolve_dataset_uri, stem_from_path
from autotune.utils import (
    cleanup,
    generate_unique_id,
    has_resumable_final_checkpoint,
    load_final_config,
    save_hpo_history,
    set_seed,
)
from autotune.validation import validate_config_for_pipeline

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random number genearator seed.",
    )

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="The YAML config file defining the hyperparameter space.",
    )

    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="A CSV, JSON or JSONL file containing the training data.",
    )

    parser.add_argument(
        "--validation_file",
        type=str,
        required=True,
        help="A CSV, JSON or JSONL file containing the validation data.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the run (will be concatenated with output_dir).",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--tuning_algo", type=str, help="Type of finetuning (e.g., alora, lora, sft).", default="none", required=False
    )

    parser.add_argument(
        "--rl_algo",
        type=str,
        help="The RL type of finetuning (e.g., dpo, orpo, grpo, ppo).",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory where models and HPO results are stored.",
    )

    parser.add_argument(
        "--output_model_name",
        type=str,
        required=True,
        help="Name of the output model (will be a subdir of the output_dir).",
    )

    parser.add_argument(
        "--ray_address",
        type=str,
        help="IP address and port of the remote ray head.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help=(
            "Number of GPU hosts in the cluster (== hosts in the LSF allocation; the Ray "
            "head colocates with a worker on host[0]). Combined with --conda_env, this "
            "triggers the multi-node blaunch launch."
        ),
    )

    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
        help="GPUs per worker node (used when --num_nodes > 1).",
    )

    parser.add_argument(
        "--conda_env",
        type=str,
        default=None,
        help="Conda env path/name workers must activate before starting Ray (required when --num_nodes > 1).",
    )

    parser.add_argument(
        "--bringup_deadline_s",
        type=int,
        default=1200,
        help=(
            "Hard deadline (seconds) for the multi-node Ray cluster bring-up "
            "(head start + blaunch'd workers + driver ray.init + worker-attach wait). "
            "If exceeded, partial teardown runs and main.py exits 0 with a warning. "
            "Default 1200 (20 minutes). Only applies on the multi-node LSF path."
        ),
    )

    parser.add_argument(
        "--fleet",
        type=str,
        choices=["a100", "h100"],
        default="a100",
        help=(
            "Deployment fleet. Selects the GPU model and the per-fleet default "
            "for NCCL_IB_HCA. 'a100' (default): A100, single rail (mlx5_0). "
            "'h100' (H100, 8-rail): H100, 8 compute rails (mlx5_0..mlx5_7). "
            "Override the rails with --ib_hca for ad-hoc benchmarking."
        ),
    )

    parser.add_argument(
        "--ib_hca",
        type=str,
        default=None,
        help=(
            "Explicit override for NCCL_IB_HCA (e.g. 'mlx5_0,mlx5_1'). "
            "Takes precedence over --fleet's default. Verify the result via "
            "ibstat in the worker pre-flight log."
        ),
    )

    parser.add_argument(
        "--ib_ifname",
        type=str,
        default=None,
        help=(
            "Optional override for NCCL_SOCKET_IFNAME (the TCP interface "
            "NCCL uses for bootstrap, before RDMA kicks in). Default unset "
            "lets NCCL auto-pick the routable Ethernet interface."
        ),
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help=(
            "Resume the final training round from the last checkpoint under "
            "<output_dir>/final_checkpoints/. When a saved final_config.json and "
            "a checkpoint exist there, HPO is skipped and the saved config drives "
            "the resumed final training. If nothing is found to resume, a warning "
            "is logged and the normal flow runs from scratch. Only applies to the "
            "multi-GPU final training stage; ignored during HPO trials."
        ),
    )

    parser.add_argument(
        "--keep_checkpoints",
        action="store_true",
        help=(
            "Skip deletion of intermediate checkpoints and training artifacts "
            "(final_checkpoints/, outputs/, train_results/, data_cache/) after "
            "final training completes. Useful for debugging. Has no effect during "
            "HPO trials (artifacts are never deleted there)."
        ),
    )

    parser.add_argument("--cleanup", help="Clean the ray_results folder.", action="store_true")

    parser.add_argument("--save_history", help="Save the HPO trial history.", action="store_true")

    parser.add_argument(
        "--no_autotune",
        help="Disable automated finetuning (use default configuration).",
        action="store_true",
    )

    parser.add_argument(
        "--job_id",
        type=str,
        required=False,
        help="Job id from autotunex",
    )

    parser.add_argument(
        "--autotunex_server_url",
        help=(
            "Base URL of an AutoTuneX bridge server to log this run to. "
            "Bridge logging is OFF by default; provide this flag to enable it. "
            "Bridge errors never fail the run."
        ),
        type=str,
        default=None,
        required=False,
    )

    # Tokenizer customization arguments
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path or name for a custom tokenizer (defaults to model_name_or_path).",
    )
    parser.add_argument(
        "--additional_special_tokens",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="Additional special tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--additional_tokens",
        nargs="+",
        type=str,
        default=None,
        required=False,
        help="Additional tokens to add to the tokenizer vocabulary.",
    )
    parser.add_argument(
        "--pad_token",
        type=str,
        default=None,
        required=False,
        help="Override the tokenizer pad token.",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        required=False,
        help="Override the tokenizer eos token.",
    )
    parser.add_argument(
        "--bos_token",
        type=str,
        default=None,
        required=False,
        help="Override the tokenizer bos token.",
    )
    parser.add_argument(
        "--data_backend",
        type=str,
        choices=["ray_data", "arrow"],
        default=None,
        help=(
            "How datasets are loaded and tokenized in the FSDP driver. "
            "'arrow' tokenizes once on the driver and writes an Arrow IPC "
            "file that each worker memory-maps (robust, no streaming "
            "coordinator). 'ray_data' runs a distributed Ray Data pipeline "
            "and auto-shards across workers (scales tokenization; requires "
            "cluster-side object-store sizing). If omitted, the value from "
            "the YAML config is used."
        ),
    )
    parser.add_argument(
        "--ray_data_concurrency",
        type=int,
        default=None,
        help=(
            "Number of parallel map_batches tasks for ray_data tokenization. "
            "Default (auto) = floor(total_cluster_cpus) - num_workers, i.e. all "
            "CPUs not reserved by this trial's GPU workers. For large concurrent "
            "HPO sweeps, set this explicitly to avoid cross-trial oversubscription."
        ),
    )
    parser.add_argument(
        "--ray_data_num_cpus",
        type=float,
        default=None,
        help=(
            "Logical CPUs reserved per ray_data tokenize task (default 1.0). "
            "Fractional values (e.g. 0.5) allow oversubscription — more tasks "
            "per physical CPU."
        ),
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="torch",
        choices=["torch", "mlx"],
        help=(
            "Training backend. 'torch' (default) uses the HuggingFace/PyTorch drivers "
            "(CUDA or MPS). 'mlx' uses the Apple Silicon MLX backend for sft/lora/qlora "
            "(requires the optional [mlx] extra; single-device only)."
        ),
    )

    # Parse the CLI arguments
    args = parser.parse_args()
    assert args.run_name is not None

    # AutoTuneX bridge logging is OFF by default. It is enabled only when the
    # user passes --autotunex_server_url. Any bridge error degrades to a warning
    # and never fails the training run.
    bridge_enabled, base_url = resolve_bridge_settings(args.autotunex_server_url)
    autotunex = None
    handler = None

    if bridge_enabled:
        try:
            user_email = get_user_details(base_url=base_url, build_id=args.job_id)
            logger.info(f"user_email: {user_email}")
            autotunex = AutoTuneXAPI(base_url=base_url, email=user_email)

            if args.job_id is not None:
                endpoint_url = f"{base_url}/fmtune/api"
                handler = BufferedLogHandler(
                    job_id=args.job_id,
                    endpoint_url=endpoint_url,
                    flush_interval=10.0,
                )
                logging.getLogger().addHandler(handler)
                logger.info(f"handler: {handler} {args.job_id}")

                # Expose job_id and endpoint_url via env vars so that Ray worker
                # processes can create their own BufferedLogHandler instances.
                os.environ["AUTOTUNE_JOB_ID"] = args.job_id
                os.environ["AUTOTUNE_ENDPOINT_URL"] = endpoint_url
        except Exception as e:
            logger.warning(f"[AutoTune] Bridge setup failed; continuing offline: {e}")
            autotunex = None
            handler = None

    # Set the seed
    set_seed(args.seed)

    # Set the local caching directory. Results will be stored here
    # before they are synced to remote storage. This env variable is ignored
    # if `storage_path` below is set to a local directory.
    # os.environ["RAY_AIR_LOCAL_CACHE_DIR"] = args.output_dir
    # Resolve the accelerator once and configure the runtime env accordingly.
    # On CUDA this sets exactly the pre-MPS env (NCCL/vLLM/verl + RAY/TUNE);
    # on MPS/CPU it omits the CUDA-only vars and enables the MPS CPU fallback.
    accel = detect_accelerator()
    logger.info(f"[AutoTune] Accelerator: {accel.kind} (count={accel.count})")
    configure_runtime_env(accel)

    # NCCL/RDMA env only matters on CUDA multi-node launches.
    if accel.kind == "cuda":
        for _k, _v in _rdma_env().items():
            os.environ.setdefault(_k, _v)

    # Set the default object store memory proportion (CUDA only) if not already set,
    # to avoid Ray Data warnings/OOMs during distributed tokenization. On non-CUDA the
    # local cluster sizes the object store explicitly (see cluster.py), so skip it.
    if accel.kind == "cuda" and "RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION" not in os.environ:
        os.environ["RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION"] = "0.5"

    # Get tuning type
    tuning_algo = args.tuning_algo
    rl_algo = args.rl_algo  # can be None for non-RL tuning methods
    save_history = args.save_history

    # Create the main config (AutotuneConfig)
    config = AutotuneConfig()
    config.load(args.config_file)
    artifact_uri, dataset_name = resolve_dataset_uri(args.train_file)
    logger.info(f"dataset_name: {dataset_name}")
    logger.info(f"dataset_artifact_uri: {artifact_uri}")

    # Pre-register config/dataset/job with the bridge only when enabled. Any
    # failure here is logged and swallowed so training still runs.
    config_id = None
    dataset_id = None
    job_details = None
    if bridge_enabled and autotunex is not None:
        try:
            result = autotunex.bootstrap(
                {
                    "job_id": args.job_id,
                    "build_id": args.job_id,
                    "config": {
                        "name": stem_from_path(args.config_file),
                        "tuner_type": tuning_algo,
                        "rl_tuner_type": rl_algo,
                        "config_data": config.config,
                    },
                    "dataset": {"name": dataset_name, "artifact_uri": artifact_uri},
                    "job": {
                        "model": args.model_name_or_path,
                        "experiment_name": args.run_name,
                        "tuning_type": tuning_algo,
                        "seed": args.seed,
                    },
                }
            )
            config_id = result["config_id"]
            dataset_id = result["dataset_id"]
            job_details = result

            # Start buffered logging now that the job exists on the server.
            # Only attach if we did not already attach one during early setup.
            if args.job_id is not None and handler is None:
                handler = BufferedLogHandler(
                    job_id=args.job_id,
                    endpoint_url=f"{base_url}/fmtune/api",
                    flush_interval=10.0,
                )
                logging.getLogger().addHandler(handler)
        except Exception as e:
            logger.warning(f"[AutoTune] Bridge pre-registration failed; continuing: {e}")

    # CLI tokenizer overrides take precedence over YAML tokenizer_config
    for attr in (
        "tokenizer_name_or_path",
        "additional_special_tokens",
        "additional_tokens",
        "pad_token",
        "eos_token",
        "bos_token",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            config.tokenizer_config[attr] = val

    # Create the tuning pipeline (AutotunePipeline)
    pipeline = AutotunePipeline(
        tuning_algo=tuning_algo,
        rl_algo=rl_algo,
        model_name_or_path=args.model_name_or_path,
    )

    # Algorithm-aware config validation. Runs after the pipeline resolves the
    # tuning/RL algorithms (online RL normalizes tuning_algo to "none"), so we
    # only require the config sections the chosen algorithms actually consume.
    validate_config_for_pipeline(
        config,
        tuning_algo=pipeline.get_tuning_algo(),
        rl_algo=pipeline.get_rl_algo(),
    )

    # Record the selected backend and enforce it early (before Ray starts).
    config.training_config["backend"] = args.backend
    if args.backend == "mlx":
        from autotune.mlx_backend import require_mlx

        require_mlx()

    # Enforce accelerator capabilities: raise on impossible configs (QLoRA, RL,
    # multi-GPU on MPS/CPU) and auto-fix benign mismatches (attn impl,
    # max_concurrent_trials, precision) with warnings. No-op on CUDA.
    apply_platform_guards(
        training_config=config.training_config,
        tune_config=config.tune_config,
        tuning_algo=pipeline.get_tuning_algo(),
        rl_algo=pipeline.get_rl_algo(),
        accel=accel,
        backend=args.backend,
    )

    # Propagate the resolved precision to the pipeline so the optimizer's
    # precision assert and the drivers agree.
    from autotune.device import resolve_precision
    from autotune.utils import get_autotune_precision

    resolved_precision = resolve_precision(config.training_config.get("precision", "bf16"), accel, probe_autocast=False)
    pipeline.set_precision(get_autotune_precision(resolved_precision))

    # Note: No need to specify the runtime_env in ray.init()
    # in the driver script.
    logger.info("[AutoTune] Connecting to ray cluster...")
    ray_info = None  # only set for local clusters
    multinode_info = None  # only set for multi-node LSF launches
    if args.ray_address is not None:
        ray.init(address=args.ray_address, log_to_driver=True, logging_level=logging.INFO)
        logger.info(f"[AutoTune] Connected to remote ray cluster at {args.ray_address}")
        configure_ray_data_context()
        logger.warning(
            "[AutoTune] If Ray Data warns that the object store is <50%% of node RAM, "
            "the remote cluster must be restarted with --object-store-memory sized "
            "accordingly (or RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.5 set on "
            "the cluster nodes). Client-side env vars do not affect an already-running cluster."
        )
    elif args.conda_env and args.num_nodes >= 1:
        # Multi-node Ray cluster via blaunch: one multi-host bsub allocation;
        # the head colocates with a worker on host[0] and blaunch starts a
        # worker on each of the other allocated hosts.
        try:
            # Worker bring-up logs land under <output_dir>/logs/ so they
            # are co-located with the rest of the run's artifacts and get
            # wiped by cleanup(). expanduser() is needed because argparse
            # stores '~/...' literally; os.path.abspath does not expand ~.
            log_dir = os.path.join(os.path.expanduser(args.output_dir), "logs")
            multinode_info = start_multinode_ray_cluster_blaunch(
                num_workers=args.num_nodes,
                gpus_per_worker=args.gpus_per_node,
                conda_env=args.conda_env,
                bringup_deadline_s=args.bringup_deadline_s,
                rl_algo=args.rl_algo,
                fleet=args.fleet,
                ib_hca=args.ib_hca,
                ib_ifname=args.ib_ifname,
                log_dir=log_dir,
            )
        except RayUpTimeoutError as e:
            logger.warning(
                f"[AutoTune] Ray cluster bring-up did not complete within "
                f"{args.bringup_deadline_s}s. Aborting run gracefully without HPO. "
                f"Details: {e}"
            )
            sys.exit(0)
        logger.info(
            f"[AutoTune] Multi-node Ray cluster ready (blaunch): "
            f"head + {args.num_nodes} worker(s), "
            f"head={multinode_info['head_address']}, GPUs={multinode_info['num_gpus']}"
        )
        configure_ray_data_context()
    else:
        # For single-node runs, use start_local_ray_cluster. Otherwise, use ray.init()
        ray_info = start_local_ray_cluster()

        # Otherwise, on LLM.build just use ray.init()
        # ray.init()
        logger.info("[AutoTune] Started local ray cluster")

    logger.info(f"[AutoTune Driver] Ray cluster resources: {ray.cluster_resources()}")

    # Redirect stdout/stderr so bare print() calls are captured as log records.
    # setup_logging() already created a StreamHandler bound to sys.__stderr__
    # (the real fd), so the PrintLogger wrapper won't cause a re-entry loop.
    sys.stdout = PrintLogger(logger)
    sys.stderr = PrintLogger(logger)

    # Generate a unique ID for the current run
    run_id = generate_unique_id()
    logger.info(f"[AutoTune] Starting run: {run_id}")

    # Routine to extract keys from jsonl dataset
    if args.train_file.endswith(".jsonl"):
        obj = None
        with open(args.train_file, "r") as f:
            for line in f:
                if line.strip():  # skip blank lines
                    obj = json.loads(line)
                    print(list(obj.keys()))
                    break

        if obj is None:
            raise ValueError(f"No records found in JSONL train file: {args.train_file}")

        keys_list = list(obj.keys())

        config.training_config["input_column"] = keys_list[0]
        config.training_config["output_column"] = keys_list[1]

    # Override data_backend from config only when the CLI arg was explicitly
    # passed. CLI default is None, so omitting --data_backend lets the YAML
    # value (normalized by AutotuneConfig) stand.
    if args.data_backend is not None:
        config.training_config["data_backend"] = args.data_backend

    # Ray Data tokenization tuning — override only when explicitly passed; the
    # drivers auto-derive sensible defaults from cluster resources otherwise.
    if args.ray_data_concurrency is not None:
        config.training_config["ray_data_concurrency"] = args.ray_data_concurrency
    if args.ray_data_num_cpus is not None:
        config.training_config["ray_data_num_cpus"] = args.ray_data_num_cpus

    # Create the hyperparameter optimizer (AutotuneOptimizer)
    optimizer = AutotuneOptimizer(
        pipeline=pipeline,
        config=config,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        output_model_name=args.output_model_name,
        resume_from_checkpoint=args.resume_from_checkpoint,
        keep_checkpoints=args.keep_checkpoints,
        cluster_resources=ray.cluster_resources(),
        run_id=run_id,
        tuner_callbacks=[CustomLoggerCallback(job_id=args.job_id, handler=handler)] if handler else [],
    )

    # If no_autotune is set, use the default configuration
    if args.no_autotune:
        print("[AutoTune] No HPO enabled. Using default hyperparameters.")
        logger.info("[AutoTune] No HPO enabled. Using default hyperparameters.")

    # Decide whether to resume final training directly from a saved config +
    # checkpoint (skipping HPO). Requires --resume_from_checkpoint AND a prior
    # interrupted final run that left both final_config.json and a checkpoint
    # under <output_dir>/final_checkpoints/.
    resume_saved = args.resume_from_checkpoint and has_resumable_final_checkpoint(args.output_dir)
    if args.resume_from_checkpoint and not resume_saved:
        logger.warning(
            "[AutoTune] --resume_from_checkpoint set but no saved final_config.json + checkpoint "
            f"under {args.output_dir}/final_checkpoints/; running the normal flow from scratch."
        )

    # Run HPO + final training, with proper error propagation and exit codes.
    exit_code = 0
    try:
        if resume_saved:
            # Resume the final training round from the saved config + checkpoint.
            # HPO is skipped entirely; the driver's _resolve_resume_checkpoint
            # picks up the last checkpoint in final_checkpoints/.
            logger.info("[AutoTune] Resuming final training from saved config; skipping HPO.")
            results = None
            result_grid = optimizer.fit_best_config(saved_config=load_final_config(args.output_dir))
        else:
            # Run HPO in a distributed manner (unless --no_autotune).
            results = optimizer.fit() if args.no_autotune is False else None

            # Save the HPO trial history
            if save_history and results is not None:
                save_hpo_history(
                    result_grid=results,
                    output_dir=args.output_dir,
                    run_name=args.run_name,
                    metric="loss",
                    mode="min",
                )

            # Train the best or the default configuration (if any)
            logger.info("[AutoTune] Training best/default config...")
            result_grid = optimizer.fit_best_config(
                use_default=args.no_autotune,
            )
        best_result = result_grid.get_best_result(metric="loss", mode="min")

        logger.info("[AutoTune] Finished training best/default config.")
        logger.info(f"[AutoTune] Tuned model score: {best_result.metrics}")
        logger.info("[AutoTune] Finished AutoTune run.")

        # AutoTune does NOT report a job-level terminal status. It runs as one
        # step of a multi-step granite.build build — a later step can fail after
        # training succeeds — so the job's terminal status is owned by the build
        # outcome (AutoTuneX's reconcile loop maps the build's final state), not
        # by AutoTune. Reporting COMPLETED/ERROR here would mark the job terminal
        # prematurely and mask the build's real result. Trial-level status (see
        # tuner_callback.py) is unaffected, and a run failure still propagates to
        # the build via the non-zero process exit code set below.
    except Exception as e:
        logger.error(f"[AutoTune] Training failed: {e}", exc_info=True)
        exit_code = 1

    finally:
        # Tear Ray down BEFORE cleanup() so teardown can write its final
        # log lines to <output_dir>/logs/... before cleanup() deletes that
        # tree. (cleanup also wipes ray_results / train_results, which Ray
        # could in principle still write to during shutdown.)
        if multinode_info is not None:
            stop_multinode_ray_cluster(multinode_info)
        elif ray_info is not None:
            stop_local_ray_cluster(ray_info["temp_dir"])
        else:
            ray.shutdown()

        if args.cleanup:
            logger.info("[AutoTune] Cleaning up the run...")
            cleanup(args.output_dir)

        logger.info("[AutoTune] Disconnected from ray cluster")

    logger.info(f"[AutoTune] Done (exit code {exit_code}).")
    # Flush stdio before any forced exit so the LSF -o file is complete.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass
    except Exception:
        pass
    if multinode_info is not None:
        # Ray's atexit hooks / gRPC client threads can keep the driver process
        # alive after the cluster is down.  Bypass interpreter shutdown so the
        # LSF job actually exits.
        os._exit(exit_code)
    sys.exit(exit_code)
