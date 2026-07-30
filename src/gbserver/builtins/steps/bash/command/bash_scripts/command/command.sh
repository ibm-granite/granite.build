#!/bin/sh
# Run an arbitrary command supplied via the build.yaml step config
# (config.bash.env.LLMB_COMMAND). The step's exit status is the command's exit
# status, so a failing command (e.g. "exit 1") hard-FAILS the target.
echo "command step start: ${LLMB_COMMAND:-<no LLMB_COMMAND set>}"
exec sh -c "${LLMB_COMMAND}"
