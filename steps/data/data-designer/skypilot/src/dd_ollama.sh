#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# A local Ollama model server for the data-designer step, so a build can generate with
# no API key and no network beyond the downloads: Data Designer calls
# http://127.0.0.1:<port>/v1 on the same node. Used only when
# data_designer_config.local_model.runtime is "ollama". Shipped to the node by
# `file_mounts: {src: src}`.
#
# CONTRACT
#   dd_ollama.sh install --version <v> --model <name> --cache-dir <dir> [--port <p>]
#       (setup) Make Ollama <v> and model <name> available under <dir>. Idempotent:
#       a finished install or an already pulled model is reused.
#   dd_ollama.sh run --version <v> --model <name> --cache-dir <dir> [--port <p>] -- <cmd...>
#       (run) Start the server on 127.0.0.1:<port>, wait until it serves <name>, run
#       <cmd...>, then stop the server. Exits with <cmd>'s status.
#
# WHAT IS DOWNLOADED, AND WHY ONLY PART OF IT IS KEPT
# The official Linux bundle (ollama-linux-<arch>.tar.zst) is ~1.5 GB, but ~1.4 GB of it
# is CUDA libraries under lib/ollama/cuda_v12 and lib/ollama/cuda_v13 (measured, v0.40.7
# arm64). The step requests no GPU, so it streams the archive and keeps only bin/ollama
# and the top-level lib/ollama/*.so — about 45 MB on disk. The download itself cannot be
# shortened: zstd is not seekable, so the whole stream is read once. The extracted result
# is cached per version and architecture, so with a shared cache that read happens once
# per cluster rather than once per run.
#
# DECOMPRESSING
# `zstd` is used when the node has it. The local Docker SLURM nodes do not, so it falls
# back to Python's `zstandard` module: $PYTHON (the step's venv, which dd_setup.sh
# installs `zstandard` into when local_model is set), else python3.
#
# THE CACHE
# <dir> should be on the environment's shared filesystem ($GB_SHARED_WORKDIR) so the
# binary and the model are fetched once and reused by later runs. Installs go to a
# temporary directory that is renamed into place only when complete, so a run never
# sees half an install.
set -euo pipefail

mode="${1:-}"; shift || true
version=""
model=""
cache_dir=""
port="11434"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --version)   version="$2";   shift 2 ;;
    --model)     model="$2";     shift 2 ;;
    --cache-dir) cache_dir="$2"; shift 2 ;;
    --port)      port="$2";      shift 2 ;;
    --)          shift; break ;;
    *)           break ;;
  esac
done

: "${version:?--version is required}"
: "${model:?--model is required}"
: "${cache_dir:?--cache-dir is required}"

case "$(uname -m)" in
  x86_64|amd64)  arch=amd64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) echo "data-designer: ERROR no Ollama build for architecture $(uname -m)" >&2; exit 1 ;;
esac

install_dir="$cache_dir/ollama-$version-$arch"
models_dir="$cache_dir/ollama-models"
ollama="$install_dir/bin/ollama"
url="https://github.com/ollama/ollama/releases/download/v$version/ollama-linux-$arch.tar.zst"

export OLLAMA_HOST="127.0.0.1:$port"
export OLLAMA_MODELS="$models_dir"

# Decompress a .tar.zst on stdin into $1, keeping bin/ollama and lib/ollama/*.so only.
extract_cpu_only() {
  local dest=$1
  if command -v zstd >/dev/null 2>&1; then
    zstd -dc | tar -x -C "$dest" --wildcards 'bin/ollama' 'lib/ollama/*' \
      --exclude='lib/ollama/*/*'
  else
    # The program goes in -c, not a heredoc on stdin: stdin is the archive.
    "${PYTHON:-python3}" -c '
import sys, tarfile
try:
    import zstandard
except ImportError:
    sys.exit("data-designer: ERROR decompressing Ollama needs zstd or the Python "
             "zstandard module; neither is available")
dest = sys.argv[1]
def wanted(name):
    if name == "bin/ollama":
        return True
    parts = name.split("/")
    return len(parts) == 3 and parts[:2] == ["lib", "ollama"]  # top level of lib/ollama only
reader = zstandard.ZstdDecompressor().stream_reader(sys.stdin.buffer)
with tarfile.open(fileobj=reader, mode="r|") as tar:
    for member in tar:
        if wanted(member.name):
            tar.extract(member, dest, filter="tar")
' "$dest"
  fi
}

install_ollama() {
  if [ -x "$ollama" ]; then
    echo "data-designer: Ollama $version ($arch) already installed in $install_dir"
    return
  fi
  mkdir -p "$cache_dir"
  local tmp
  tmp="$(mktemp -d "$cache_dir/.ollama-install.XXXXXX")"
  echo "data-designer: installing Ollama $version ($arch, CPU libraries only) into $install_dir"
  curl -fsSL "$url" | extract_cpu_only "$tmp"
  [ -x "$tmp/bin/ollama" ] || { echo "data-designer: ERROR $url had no bin/ollama" >&2; exit 1; }
  # Another run may have finished the same install meanwhile; keep whichever won.
  mv -T "$tmp" "$install_dir" 2>/dev/null || rm -rf "$tmp"
}

server_pid=""
start_server() {
  mkdir -p "$models_dir"
  "$ollama" serve >"${OLLAMA_LOG:-./ollama-serve.log}" 2>&1 &
  server_pid=$!
  local _
  for _ in $(seq 1 60); do
    if curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then return; fi
    kill -0 "$server_pid" 2>/dev/null || {
      echo "data-designer: ERROR the Ollama server exited during startup:" >&2
      tail -20 "${OLLAMA_LOG:-./ollama-serve.log}" >&2; exit 1; }
    sleep 1
  done
  echo "data-designer: ERROR the Ollama server did not answer on $OLLAMA_HOST within 60s" >&2
  exit 1
}

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap stop_server EXIT

case "$mode" in
  install)
    install_ollama
    start_server
    echo "data-designer: pulling $model into $models_dir"
    "$ollama" pull "$model"
    stop_server
    ;;
  run)
    [ "$#" -gt 0 ] || { echo "data-designer: ERROR run needs a command after --" >&2; exit 1; }
    [ -x "$ollama" ] || { echo "data-designer: ERROR Ollama is not installed in $install_dir (setup did not run?)" >&2; exit 1; }
    start_server
    # Pulled in setup; a run on a warm cluster may reach here without it, so pull if absent.
    "$ollama" show "$model" >/dev/null 2>&1 || "$ollama" pull "$model"
    echo "data-designer: serving $model on http://$OLLAMA_HOST/v1"
    set +e
    "$@"
    status=$?
    set -e
    stop_server
    exit "$status"
    ;;
  *)
    echo "data-designer: ERROR usage: dd_ollama.sh install|run --version V --model M --cache-dir D [--port P] [-- cmd...]" >&2
    exit 2
    ;;
esac
