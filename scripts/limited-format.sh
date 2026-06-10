#!/usr/bin/env bash

MY_LINE_BREAK="---------------------------"

echo 'format start'

# check changed files and
# collect the relative file paths in an array
 mapfile -td '' files < <(git diff main...HEAD --name-only -z --format=)
#mapfile -td '' files < <(git diff dev...HEAD --name-only -z --format=)

# run formatter separately for each file
for x in "${files[@]}" ;
do
    if [[ "${x}" = *.py ]]; then
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
