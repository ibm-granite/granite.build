#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Config guards for the eval/lm-eval step, called at the top of both the `setup:`
# and `run:` blocks of the generated step.yaml (shipped by `file_mounts: {src: src}`).
#
# Both phases, as in dpk: `setup` is the expensive one (the torch and lm-eval install),
# so a bad config is refused before it is paid for; `run` can be reached on a warm
# cluster without a fresh setup.
#
# Keys arrive as DATA, one option per key, rather than being read back out of the
# rendered lm_eval argv: a key containing a space stays a single word, and an empty
# key is still visible.
#
# CONTRACT
#   lm_eval_guard.sh --backend <b> --install <true|false> --version <v>
#                    [--task <t>]...  [--config-key <k>]...
#                    [--model-arg-key <k>]...  [--arg-key <k>]...  [--]
set -euo pipefail

P="eval/lm-eval"
backend=""
install=""
version=""
tasks=()
config_keys=()
model_arg_keys=()
arg_keys=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --backend)       backend="$2";             shift 2 ;;
    --install)       install="$2";             shift 2 ;;
    --version)       version="$2";             shift 2 ;;
    --task)          tasks+=("$2");            shift 2 ;;
    --config-key)    config_keys+=("$2");      shift 2 ;;
    --model-arg-key) model_arg_keys+=("$2");   shift 2 ;;
    --arg-key)       arg_keys+=("$2");         shift 2 ;;
    --)              shift; break ;;
    *)echo "$P: ERROR internal: unknown guard option '$1'" >&2; exit 2 ;;
  esac
done

fail() { echo "$P: ERROR $*" >&2; exit 1; }

# Every key the step reads. Anything else under lm_eval_config is a typo or a key
# from another step, and would otherwise be silently ignored.
KNOWN=" image install lm_eval_version extras extra_packages torch_index_url"
KNOWN+=" pip_index_url backend tasks model_args limit num_fewshot batch_size device"
KNOWN+=" gen_kwargs apply_chat_template log_samples trust_remote_code seed args "

for k in ${config_keys[@]+"${config_keys[@]}"}; do
  case "$k" in
    model|model_path|pretrained|model_dir)
      fail "lm_eval_config.$k is not read: the model is the step's required 'model' input. Bind it on the target (inputs: model: ...) and remove lm_eval_config.$k." ;;
    output_path|output_dir|results_path)
      fail "lm_eval_config.$k is not read: results go to the step's 'results' output. Declare it on the target (outputs: results: ...) and remove lm_eval_config.$k." ;;
  esac
  case "$KNOWN" in
    *" $k "*) ;;
    *) fail "unknown key lm_eval_config.$k (known keys:${KNOWN% })" ;;
  esac
done

# Backends whose model_args take `pretrained=<path>`, which is how the bound model
# input reaches lm_eval. API backends (local-completions, openai-*) take a URL instead
# and are not supported by this step.
case "$backend" in
  hf|vllm) ;;
  *) fail "lm_eval_config.backend must be 'hf' or 'vllm' (got '$backend')." ;;
esac

if [ "$install" = "true" ] && [ -z "$version" ]; then
  fail "lm_eval_config.lm_eval_version is empty; pin a version (e.g. \"0.4.13\") or set install: false with an image that provides lm_eval."
fi

if [ "${#tasks[@]}" -eq 0 ]; then
  fail "lm_eval_config.tasks is empty; name at least one task (e.g. [\"aime25\"]). 'lm-eval ls tasks' lists them."
fi
for t in "${tasks[@]}"; do
  [ -n "$t" ] || fail "lm_eval_config.tasks contains an empty task name."
  case "$t" in
    *[[:space:]]*|*,*) fail "lm_eval_config.tasks: '$t' is not one task name; give a list, one task per item." ;;
  esac
done

for k in ${model_arg_keys[@]+"${model_arg_keys[@]}"}; do
  [ "$k" = "pretrained" ] && fail "lm_eval_config.model_args.pretrained is set by the step from the bound 'model' input; remove it."
  [[ "$k" =~ ^[A-Za-z_][A-Za-z0-9_.-]*$ ]] || fail "lm_eval_config.model_args: '$k' is not a valid key."
done

# Flags the step owns: the argv shape, the output location, and every flag that has
# its own lm_eval_config key (one place to set each thing).
OWNED=" model model_args tasks output_path config limit num_fewshot batch_size device"
OWNED+=" gen_kwargs apply_chat_template log_samples trust_remote_code seed "
for k in ${arg_keys[@]+"${arg_keys[@]}"}; do
  [[ "$k" =~ ^[a-z][a-z0-9_]*$ ]] || fail "lm_eval_config.args: '$k' is not a flag name (write it without dashes, e.g. system_instruction)."
  case "$OWNED" in
    *" $k "*)
      case "$k" in
        model|model_args|tasks|output_path|config)
          fail "lm_eval_config.args.$k is set by the step; remove it." ;;
        *)
          fail "lm_eval_config.args.$k: use lm_eval_config.$k instead." ;;
      esac ;;
  esac
done
