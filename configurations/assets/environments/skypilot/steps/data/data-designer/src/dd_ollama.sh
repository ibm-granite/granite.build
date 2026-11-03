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
# makes Python 3.12 and installs `zstandard` into when local_model is set), else
# python3. 3.12+ because extraction uses tarfile's `filter="tar"`.
#
# VERIFYING THE DOWNLOAD
# The archive is checked against a known sha256 before anything from it is installed,
# and the result is cached and reused by later builds, so an unverified download must
# never reach the cache. The archive is saved to a temporary file first, hashed, then
# extracted: verifying a stream while extracting it would install files before the
# hash is known.
#
# For the step's default release the hashes are PINNED below, copied from that
# release's sha256sum.txt. That protects against a tampered release, not only a
# corrupt download: a hash fetched from the same place as the archive would agree
# with a replaced archive (third review). For any other --version the script falls
# back to the published sha256sum.txt, which still catches corruption.
#
# THE CACHE
# <dir> should be on the environment's shared filesystem ($GB_SHARED_WORKDIR) so the
# binary and the model are fetched once and reused by later runs. Installs go to a
# temporary directory that is renamed into place only when complete, so a run never
# sees half an install, and the temporary directory is removed if the install fails.
# Model pulls are serialized with a lock on the cache, since two servers pulling into
# the same model store at once could corrupt it.
#
# THE PORT
# On a shared node another job may already serve on the same port. A readiness check
# alone would then succeed against THAT server, and this run would generate through
# another job's model, losing it when that job ends (found in review). So the server
# must be our own process: if it exits during startup, the port is taken, and the run
# fails with that reason rather than borrowing the other server.
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
# The model store is shared by every build on the cluster. By default `ollama serve`
# deletes unreferenced blobs at startup, which would remove the partial blobs of a pull
# another build is still running; disable that. (Pulls themselves are serialized with
# flock below.)
export OLLAMA_NOPRUNE=1

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

# Pinned sha256 of ollama-linux-<arch>.tar.zst for the step's default release, from
# https://github.com/ollama/ollama/releases/download/v0.40.7/sha256sum.txt. Update both
# together with local_model.version's default in step-template.yaml.
pinned_sha256() {
  case "$version/$arch" in
    0.40.7/amd64) echo 7f082e625b4c334df84ed1558fe282d6d8091fe3545790aef30b3a9c50d44d41 ;;
    0.40.7/arm64) echo c74249e87795b1437ce0f484adfd05b4628d7497472c475b13ec6db69ecc69cf ;;
  esac
}

# The sha256 to verify against: pinned for the default release, otherwise the one
# published in that release's sha256sum.txt. Prints nothing when neither is available.
expected_sha256() {
  local pinned
  pinned="$(pinned_sha256)"
  if [ -n "$pinned" ]; then echo "$pinned"; return; fi
  curl -fsSL "https://github.com/ollama/ollama/releases/download/v$version/sha256sum.txt" \
    | awk -v f="./ollama-linux-$arch.tar.zst" '$2 == f || $2 == substr(f, 3) {print $1; exit}'
}

# sha256 of a file. GNU coreutils' sha256sum on Linux nodes; shasum where that is all
# there is (macOS, which the unit tests run on).
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  else shasum -a 256 "$1" | awk '{print $1}'; fi
}

install_tmp=""
cleanup_install() { [ -z "$install_tmp" ] || rm -rf "$install_tmp"; }

install_ollama() {
  if [ -x "$ollama" ]; then
    echo "data-designer: Ollama $version ($arch) already installed in $install_dir"
    return
  fi
  mkdir -p "$cache_dir"
  install_tmp="$(mktemp -d "$cache_dir/.ollama-install.XXXXXX")"
  echo "data-designer: installing Ollama $version ($arch, CPU libraries only) into $install_dir"

  local want got
  # `|| true`: under `set -e -o pipefail` a failed fetch would otherwise exit inside the
  # command substitution, skipping the explanation below (third review).
  want="$(expected_sha256 || true)"
  [ -n "$want" ] || {
    echo "data-designer: ERROR could not get a sha256 for ollama-linux-$arch.tar.zst" \
         "of Ollama $version (not pinned, and its sha256sum.txt could not be fetched" \
         "or has no entry); nothing was installed" >&2
    exit 1; }
  curl -fsSL -o "$install_tmp/archive.tar.zst" "$url"
  got="$(sha256_of "$install_tmp/archive.tar.zst")"
  if [ "$got" != "$want" ]; then
    echo "data-designer: ERROR the Ollama $version archive failed its checksum" \
         "(expected $want, got $got); nothing was installed" >&2
    exit 1
  fi
  mkdir -p "$install_tmp/root"
  extract_cpu_only "$install_tmp/root" <"$install_tmp/archive.tar.zst"
  rm -f "$install_tmp/archive.tar.zst"
  [ -x "$install_tmp/root/bin/ollama" ] || { echo "data-designer: ERROR $url had no bin/ollama" >&2; exit 1; }
  # Rename into place. A plain `mv src dest` with no existing dest is an atomic rename
  # on one filesystem. If another run finished the same install meanwhile, dest exists:
  # `mv` would then nest src INSIDE it, so check first and keep the one that won.
  # (Portable on purpose: `mv -T` is GNU-only, and its failure used to be swallowed.)
  # The EXIT trap's cleanup_install removes install_tmp (and root/ with it) either way.
  [ -e "$install_dir" ] || mv "$install_tmp/root" "$install_dir" 2>/dev/null || true
  cleanup_install
  install_tmp=""
  [ -x "$ollama" ] || { echo "data-designer: ERROR could not install Ollama into $install_dir" >&2; exit 1; }
}

# Serialize model pulls into the shared store: two servers pulling the same model into
# one OLLAMA_MODELS at once could leave a corrupt blob for every later build.
# flock(1) is util-linux, present on the SLURM nodes and in SkyPilot's kube image; where
# it isn't (macOS, where the unit tests run), the pull simply runs unlocked.
pull_model() {
  mkdir -p "$models_dir"
  (
    if command -v flock >/dev/null 2>&1; then flock 9; fi
    "$ollama" show "$model" >/dev/null 2>&1 || "$ollama" pull "$model"
  ) 9>"$cache_dir/.ollama-pull.lock"
}

server_pid=""
port_taken() {
  echo "data-designer: ERROR port ${port} on this node is already in use (another job's" \
       "server?). Set data_designer_config.local_model.port to a free port." >&2
  exit 1
}

start_server() {
  mkdir -p "$models_dir"
  local log="${OLLAMA_LOG:-./ollama-serve.log}"
  # 1. Before starting: anything already answering on the port belongs to someone else.
  if curl -fs --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
    port_taken
  fi
  : >"$log"
  "$ollama" serve >"$log" 2>&1 &
  server_pid=$!
  # 2. Readiness is OUR server saying it is listening on our address, not merely an answer
  #    on the port: another job's server can answer while ours is still failing to bind,
  #    which a "process alive + port answers" check accepted (second review, reproduced
  #    with a server that exits 0.5s after start). Ollama logs
  #    `Listening on 127.0.0.1:<port>` once bound, and
  #    `bind: address already in use` when the port is taken (both checked against a
  #    real `ollama serve`).
  local _
  for _ in $(seq 1 120); do
    if grep -qF "Listening on $OLLAMA_HOST" "$log" 2>/dev/null \
       && kill -0 "$server_pid" 2>/dev/null \
       && curl -fs --max-time 2 "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      server_pid=""
      grep -qiF "address already in use" "$log" 2>/dev/null && port_taken
      echo "data-designer: ERROR the Ollama server exited during startup:" >&2
      tail -20 "$log" >&2
      exit 1
    fi
    sleep 0.5
  done
  echo "data-designer: ERROR the Ollama server did not start listening on $OLLAMA_HOST within 60s" >&2
  tail -20 "$log" >&2
  exit 1
}

stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
  server_pid=""
}
trap 'stop_server; cleanup_install' EXIT

case "$mode" in
  install)
    install_ollama
    start_server
    echo "data-designer: pulling $model into $models_dir"
    pull_model
    stop_server
    ;;
  run)
    [ "$#" -gt 0 ] || { echo "data-designer: ERROR run needs a command after --" >&2; exit 1; }
    [ -x "$ollama" ] || { echo "data-designer: ERROR Ollama is not installed in $install_dir (setup did not run?)" >&2; exit 1; }
    start_server
    # Pulled in setup; a run on a warm cluster may reach here without it, so pull if absent.
    pull_model
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
