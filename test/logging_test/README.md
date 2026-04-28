### Logging tests required harness on Vela
#### log-generator.yaml
To deploy, run `oc apply -f log-generator.yaml` after logging on to the Vela cluster. The deployment
submits an AppWrapper named gb-logging-gen to the default-queue for scheduling. The number of test
log messages and their emitting frequency are set in Line 26. 

The deployment ends automatically after the number of emitted log messages has been reached. However,
to terminate the deployment early. run `oc delete -f log-generator.yaml`