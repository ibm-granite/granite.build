# NeMo-Skills family image setup (runs as root during the build; README.md).
# Everything fetched here is pinned by commit and/or sha256 and verified.

fetch() {  # fetch <out> <sha256> <url>...: the first URL whose content matches
    local out=$1 sha=$2 url
    shift 2
    for url in "$@"; do
        rm -f "$out.part"
        if curl -fL --retry 3 --retry-delay 5 -o "$out.part" "$url" \
            && echo "$sha  $out.part" | sha256sum -c -; then
            mv "$out.part" "$out"
            return 0
        fi
        echo "fetch: $url did not give $sha, trying the next source" >&2
    done
    rm -f "$out.part"
    echo "fetch: no source gave $out with sha256 $sha" >&2
    return 1
}

# ns's RULER prepare checks for git-lfs; unzip for the nltk data.
apt-get update
apt-get install -y --no-install-recommends git-lfs unzip

# --- SciCode (scicode): ns's sandbox server env and test data ---------------------
# ns's sandbox image (dockerfiles/Dockerfile.sandbox @ ns bcf059af) runs Python 3.10
# with requirements/sandbox.lock, and its SciCode evaluator then force-installs
# scipy==1.10.1 (which takes numpy to 1.26.4). This env is that end state for the
# packages SciCode and the server use, at the lock's versions. `pip` in it is a
# no-op, so the evaluator's own pip calls cannot change it at run time.
export UV_PYTHON_INSTALL_DIR=/opt/uv-python
# The build runs this from /opt/granite-evals, whose pyproject.toml overrides numpy>=2
# ([tool.uv] override-dependencies): uv pip would apply that here too, and
# scipy 1.10.1 cannot load under numpy 2. These envs take no project config.
export UV_NO_CONFIG=1
uv venv --python 3.10 /opt/ns-sandbox
uv pip install --python /opt/ns-sandbox/bin/python \
    flask==3.1.3 werkzeug==3.1.8 ipython==8.39.0 traitlets==5.14.3 psutil==7.2.2 \
    numpy==1.26.4 scipy==1.10.1 sympy==1.14.0 mpmath==1.3.0 h5py==3.16.0 matplotlib==3.10.8
cat > /opt/ns-sandbox/bin/pip <<'EOF'
#!/bin/sh
echo "granite: pip is disabled in the SciCode sandbox (its packages are pinned in the image): pip $*" >&2
exit 0
EOF
chmod 755 /opt/ns-sandbox/bin/pip
uv pip freeze --python /opt/ns-sandbox/bin/python > /opt/ns-sandbox/granite-freeze.txt
chmod -R a+rX /opt/uv-python /opt/ns-sandbox

# SciCode's test targets, where ns's evaluator reads them. The Google Drive file ns's
# sandbox image downloads; two HF copies (same sha256) as fallbacks.
mkdir -p /data
fetch /data/test_data.h5 48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890 \
    "https://drive.usercontent.google.com/download?id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW&export=download&confirm=t" \
    "https://huggingface.co/datasets/Srimadh/Scicode-test-data-h5/resolve/72c247d3a8410921b2e848e046d71ed63d9a0ddb/test_data.h5" \
    "https://huggingface.co/datasets/Innovator-Evaluation/scicode_h5py_file/resolve/dca99b73bea8d63a41b11446f29e3e82db7d948c/test_data.h5"
chmod 644 /data/test_data.h5

# --- RULER (ruler-*): the generator scripts and their source data -----------------
RULER_COMMIT=c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
git init -q /opt/ruler
git -C /opt/ruler fetch -q --depth 1 https://github.com/NVIDIA/RULER "$RULER_COMMIT"
GIT_LFS_SKIP_SMUDGE=1 git -C /opt/ruler checkout -q FETCH_HEAD
rm -rf /opt/ruler/.git
echo "$RULER_COMMIT" > /opt/ruler/GRANITE_EVALS_COMMIT
J=/opt/ruler/scripts/data/synthetic/json
fetch "$J/english_words.json" affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca \
    "https://media.githubusercontent.com/media/NVIDIA/RULER/$RULER_COMMIT/scripts/data/synthetic/json/english_words.json"
fetch "$J/squad.json" 80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8 \
    "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
fetch "$J/hotpotqa.json" e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8 \
    "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json" \
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"
# The essays: RULER's own download script (49 from gkamradt's repo, the rest from
# paulgraham.com), checked against the sha256 of the pinned result.
uv venv --python /usr/local/bin/python3 /tmp/pg-venv
uv pip install --python /tmp/pg-venv/bin/python html2text==2025.4.15 beautifulsoup4==4.15.0 tqdm==4.67.1
(cd "$J" && /tmp/pg-venv/bin/python download_paulgraham_essay.py | tail -3)
echo "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd  $J/PaulGrahamEssays.json" | sha256sum -c -
rm -rf /tmp/pg-venv "$J/essay_repo" "$J/essay_html"
chmod -R a+rX /opt/ruler

# nltk's sentence tokenizer for RULER's generators, from a pinned nltk_data commit
# (checked by git blob id, then recorded by sha256).
NLTK_DATA_COMMIT=550b6625bcef1f2abff2ff770a5a0d272c9c6b2a
D=/usr/local/share/nltk_data/tokenizers
mkdir -p "$D"
for z in punkt:da7ffbd1e6fd6cc5c2f6879c2d4da23c7691944c punkt_tab:5e5ff6137d5ee6025e400d1c3a7b21914c48b635; do
    name=${z%%:*} blob=${z#*:}
    curl -fL --retry 3 -o "$D/$name.zip" \
        "https://raw.githubusercontent.com/nltk/nltk_data/$NLTK_DATA_COMMIT/packages/tokenizers/$name.zip"
    got=$( (printf 'blob %s\0' "$(stat -c %s "$D/$name.zip")"; cat "$D/$name.zip") | sha1sum | cut -d' ' -f1)
    [ "$got" = "$blob" ] || { echo "nltk $name.zip: git blob $got != $blob" >&2; exit 1; }
    sha256sum "$D/$name.zip" >> /usr/local/share/nltk_data/GRANITE_EVALS_SHA256
    unzip -q -o "$D/$name.zip" -d "$D"
done
chmod -R a+rX /usr/local/share/nltk_data

# --- WMT24++ (wmt24pp): the COMET scorer's env -------------------------------------
# ns scores translations with unbabel-comet (XCOMET-XXL), which it pip-installs
# unpinned at run time. unbabel-comet needs transformers<5, numpy<2 and protobuf<5,
# which the job venv's vLLM cannot share, so it gets its own env, hash-locked in
# comet-requirements.txt (setuptools<81: comet imports pkg_resources). The model is
# not baked in: it is gated (CC-BY-NC-SA-4.0); the run downloads it at a pinned
# revision (granite_evals.benchmarks.nemo_skills_extended.WMT24pp).
uv venv --python /usr/local/bin/python3 /opt/comet
uv pip install --python /opt/comet/bin/python --require-hashes -r /tmp/extras/comet-requirements.txt
/opt/comet/bin/python -c "import comet, torch, transformers; print('comet env', torch.__version__, transformers.__version__)"
uv pip freeze --python /opt/comet/bin/python > /opt/comet/granite-freeze.txt
chmod -R a+rX /opt/comet
