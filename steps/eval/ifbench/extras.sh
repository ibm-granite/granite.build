# IFBench's evaluator code, where NeMo-Skills' ifbench evaluator runs it
# (`cd /opt/benchmarks/IFBench && python -m run_eval`), set up the way ns's own
# image does (dockerfiles/Dockerfile.nemo-skills at the pinned ns commit): the
# IFBench commit ns pins, ns's ifbench.patch (no model download at import, a
# crashing verifier scores as not followed), and the NLTK data ns predownloads.
# Its Python requirements are the `ifbench` extra (pyproject.toml).
IFBENCH_COMMIT=c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c
NS_COMMIT=bcf059af55c20a89f797724598f9908d126153e6
NS_PATCH_SHA256=e6fb383ddae3dd9e53323ff3ef0d36218821ed37455b6b724f866b0b13dd0b0c
NLTK_DATA_COMMIT=550b6625bcef1f2abff2ff770a5a0d272c9c6b2a

dir=/opt/benchmarks/IFBench
git init -q "$dir"
git -C "$dir" remote add origin https://github.com/allenai/IFBench.git
git -C "$dir" fetch -q --depth 1 origin "$IFBENCH_COMMIT"
git -C "$dir" reset -q --hard FETCH_HEAD
curl -fsSL -o "$dir/ifbench.patch" \
    "https://raw.githubusercontent.com/NVIDIA-NeMo/Skills/${NS_COMMIT}/dockerfiles/ifbench.patch"
echo "${NS_PATCH_SHA256}  $dir/ifbench.patch" | sha256sum -c -
git -C "$dir" apply ifbench.patch
rm -rf "$dir/.git"

# NLTK data (a default nltk search path), pinned to an nltk_data commit.
nltk=/usr/local/share/nltk_data
for pkg in tokenizers/punkt tokenizers/punkt_tab corpora/stopwords taggers/averaged_perceptron_tagger_eng; do
    mkdir -p "$nltk/$(dirname "$pkg")"
    curl -fsSL -o /tmp/nltk.zip "https://raw.githubusercontent.com/nltk/nltk_data/${NLTK_DATA_COMMIT}/packages/${pkg}.zip"
    python3 -m zipfile -e /tmp/nltk.zip "$nltk/$(dirname "$pkg")"
    rm /tmp/nltk.zip
done
chmod -R a+rX /opt/benchmarks "$nltk"
