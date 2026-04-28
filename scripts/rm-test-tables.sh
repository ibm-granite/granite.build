if [ -z "$LAKEHOUSE_ENVIRONMENT" ]; then
    # Require this really only logging
    echo LAKEHOUSE_ENVIORNMENT env var must be set
    exit 1
fi
rm_test_tables() {
    ns=$1
    match=$2
    tables=$(dmf table ls -n $ns --show-access | sed -e "s/'//g" -e "s/)//g" -e 's/,//g' | awk '{print $4}' | grep '^test' | grep $2 | sort) 
    if [ -z "$tables" ]; then
	echo No test tables found to delete from namespace $ns in Lakehouse $LAKEHOUSE_ENVIRONMENT environment.
	return	
    fi
    # Log what we plan to do
    for i in $tables; do
       echo $i
    done
    echo Planning to delete the above tables from namespace $ns in Lakehouse $LAKEHOUSE_ENVIRONMENT environment.
    read -p "Hit enter to proceed, ^C otherwise."
    # Do the work
    for i in $tables; do
       z=$(echo $i | grep '^gb_')	# Double-check to protect our real tables.
       if [ -z "$z" ]; then
	   echo Deleting from namespace $ns table $i
	   dmf table delete -n $ns -t $i 
       fi
    done
}
rm_test_tables granite_dot_build.admin _gb_
rm_test_tables granite_dot_build.public test_dl_ 
