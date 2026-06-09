#!/usr/bin/env bash

MY_LINE_BREAK="---------------------------"

echo 'typecheck start'

# check changed files and
# collect the relative file paths in an array
mapfile -td '' files < <(git diff main...HEAD --name-only -z --format=)
#mapfile -td '' files < <(git diff dev...HEAD --name-only -z --format=)

# run static check
# MY_OUTPUT="$(mypy --disable-error-code=import-untyped src/gbserver/)"
MY_OUTPUT="$(make staticcheck)"

# grep for the file name in the static check output
for x in "${files[@]}" ;
do
    echo "${MY_LINE_BREAK}"
    echo -e "\033[0;36mFile: \033[0m\033[0;32m${x}\033[0m"
    echo
    echo "${MY_OUTPUT}" | grep "${x}"
done

echo "${MY_LINE_BREAK}"
echo 'typecheck end'
