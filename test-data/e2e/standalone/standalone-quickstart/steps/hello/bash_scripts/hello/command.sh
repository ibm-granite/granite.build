#!/bin/sh
echo 'hello step start'
echo "Hello from gbserver standalone!"
echo "Running on: $(hostname)"
echo "Environment: ${GBSERVER_ENVIRONMENT_TYPE:-bash}"
echo 'hello step end'
