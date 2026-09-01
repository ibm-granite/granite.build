#!/usr/bin/env bash

MY_LINE_BREAK="---------------------------"

echo 'format start'

# Fail loudly instead of formatting nothing. Without `pipefail` a failure in the
# `git diff` below is swallowed by the pipe and the loop simply iterates zero
# times, so the script prints "format start"/"format end" and exits 0 having
# formatted NOTHING — which is what a missing `mapfile` used to do silently on
# macOS's bash 3.2 (see below), letting unformatted code reach CI.
set -o pipefail

# Collect the changed files, NUL-delimited so a path containing whitespace stays
# one entry. Read with a `while` loop rather than `mapfile -d`: mapfile's `-d`
# flag needs bash 4, and macOS ships bash 3.2 (Apple froze it at the last
# GPLv2 release), where `mapfile: command not found` made this a silent no-op.
# `read -d ''` is the bash 3.2-compatible equivalent and works on bash 4+ too.
#
# The loop body runs in the CURRENT shell (the redirect is on `done`, not a pipe),
# so `files` survives afterwards.
files=()
while IFS= read -r -d '' x; do
    files+=("${x}")
done < <(git diff main...HEAD --name-only -z --format=)
# To format against `dev` instead, change `main...HEAD` above.

if [[ "${#files[@]}" -eq 0 ]]; then
    echo "no changed files vs main; nothing to format"
    echo "${MY_LINE_BREAK}"
    echo 'format end'
    exit 0
fi

# run formatter separately for each file
for x in "${files[@]}" ;
do
    # Skip paths that no longer exist on disk. `git diff` lists deleted and
    # renamed-away files, but formatting them would make isort/black fail on a
    # missing path (isort falls back to treating it as an invalid settings dir).
    if [[ ! -f "${x}" ]]; then
        echo "skip missing file: ${x}"
        continue
    fi
    if [[ "${x}" = *.py ]]; then
        # Honour the repo-wide formatter exclusions. isort's `extend_skip_glob`
        # and black's `extend-exclude` (both set for autotunex/ in the root
        # pyproject.toml) are applied while WALKING a directory — naming a file
        # explicitly, as this script does, bypasses them and reformats the file
        # anyway. autotunex/ is ruff-formatted at line-length 100 with its own CI,
        # so running repo-root black over it reflows ~260 files that are not the
        # author's to change. Skip those paths here instead.
        if [[ "${x}" = autotunex/* ]]; then
            echo "skip excluded path (own formatter/CI): ${x}"
            continue
        fi
        echo "${MY_LINE_BREAK}"
        echo -e "\033[0;36m Formatting file: \033[0m\033[0;32m${x}\033[0m"
        isort --profile black "${x}"
        black "${x}"
        echo
    else
        echo "skip non-python file: ${x}"
    fi
done

echo "${MY_LINE_BREAK}"
echo 'format end'
