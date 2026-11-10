# τ³-bench task data: data/tau2 of the tau2-bench commit the `tau` extra pins
# (the wheel has no data). Same layout as granite_evals.benchmarks.tau.fetch_data:
# /opt/tau2-data/<commit>/{data,.granite-commit}.
COMMIT=fc0055dc4e0a316c3f83133267fbd6faaa770992
DEST=/opt/tau2-data/$COMMIT
mkdir -p "$DEST"
cd "$DEST"
git init -q
git remote add origin https://github.com/sierra-research/tau2-bench
git sparse-checkout set data/tau2/domains data/tau2/user_simulator
git fetch -q --depth 1 --filter=blob:none origin "$COMMIT"
git checkout -q FETCH_HEAD
test "$(git rev-parse HEAD)" = "$COMMIT"
rm -rf .git
echo "$COMMIT" > .granite-commit
chmod -R a+rX /opt/tau2-data
du -sh "$DEST"
