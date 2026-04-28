# A script that assumes you are oc logged in and then captures all logs from all gb-build-runner pods
# Creates gb-build-runner-<build-id>-<hash>.txt files
# Usage: get-runner-logs.sh
# Hit ^C to terminate this script.
while true; do
  runners=$(oc get pods | grep runner | grep Running | grep -v '[1-9]h' | awk '{print $1}')
  for i in $runners; do
    log=$i.txt
    touch $log
    tmp=tmp$$
    oc logs $i > $tmp
    len=$(wc -l $log | awk '{print $1}') 
    newlen=$(wc -l $tmp| awk '{print $1}') 
    if [ $newlen -gt $len ]; then
      mv $tmp $log 
    fi
    rm -f $tmp
  done
  sleep 2
done
