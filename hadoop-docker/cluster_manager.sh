#!/usr/bin/env bash

usage="$(basename "$0")"
usage=$usage' [-h | --help] -- manage containers for a stock prediction cluster
	[-i -k -c -d -s -t -b] [--init --kill --create --destroy --start --terminate --build] 

	-i: 	initialize containers
	-k: 	kill all containers
	-c:	create a subnetwork for the stockprediction cluster
	-d:  	destroy the subnetwork
	-s:	start the docker system service
	-t:	terminate the docker system service
	-b:	build the images required to initialize instances of the containers
'
OPTS=`getopt -o hikcdstb -l help,init,kill,create,destroy,start,terminate,build -- "$@"`
eval set -- "$OPTS"

TASK="FALSE"

# define vars for hadoop
HADOOP_HOME="/usr/local/hadoop"
HADOOP_COMMON_HOME="/usr/local/hadoop"
HADOOP_HDFS_HOME="/usr/local/hadoop"
HADOOP_MAPRED_HOME="/usr/local/hadoop"
HADOOP_YARN_HOME="/usr/local/hadoop"
HADOOP_CONF_DIR="/usr/local/hadoop/etc/hadoop"
YARN_CONF_DIR="$HADOOP_HOME/etc/hadoop"

while true; do
	case "$1" in
		"-h" | "--help") 		echo "$usage"
		                 		shift; exit;;
		"-i" | "--init") 		TASK="INIT"; shift;;
		"-k" | "--kill") 		TASK="KILL"; shift;;
		"-c" | "--create")		TASK="CREATE"; shift;;
		"-d" | "--destroy") 		TASK="DESTROY"; shift;;
		"-s" | "--start")		TASK="START"; shift;;
		"-t" | "--terminate")		TASK="TERMINATE"; shift;;
		"-b" | "--build")		TASK="BUILD"; shift;;
		-- )  shift; break;;
		*) break;;
	esac
done

############################################################
### Functions:

# Initializes desired amount of containers
init_containers() {

	docker run -d --net hadoop --net-alias yarnmaster  --name yarnmaster -h yarnmaster -p 8032:8032 -p 8088:8088 jbuxofplenty/yarnmaster-stockprediction
	docker run -d --net hadoop --net-alias namenode --name namenode -h namenode -p 8020:8020 jbuxofplenty/namenode-stockprediction
	
	for ((i=1;i<=10;i++)); 
	do 
		docker run -d --net hadoop --net-alias datanode_$i  -h datanode_$i --name datanode_$i --link namenode --link yarnmaster jbuxofplenty/datanode-stockprediction
	done
	
	# Copy the stockprediction files over to the distributed file system
	docker exec -ti -u root namenode bash -c "hdfs dfs -mkdir /stockprediction"
        docker exec -ti -u root namenode bash -c "hdfs dfs -put /tmp/stockprediction/* /stockprediction/"

}

# Kills desired amount of containers
kill_containers() {
	# Copy the distributed stockprediction files back to the local filesystem of the namenode docker instance
        docker exec -ti -u root namenode bash -c "rm -rf /tmp/stockprediction/*"
	docker exec -ti -u root namenode bash -c "hdfs dfs -get /stockprediction/. /tmp/"	
	
	# Delete the old backup
	rm -rf backup/stockprediction

	# Pull the stockprediction data and store it on the local filesystem that's running this script	
	docker cp namenode:/tmp/stockprediction/. backup/stockprediction/

	# Shutdown all of the processes before removing the containers
	docker exec -di -u root namenode bash -c "$HADOOP_PREFIX/sbin/stop-all.sh" & 
	
	# Allow some time for the hadoop cluster to tear down all of the processes
	sleep 2

	# Remove the manager containers
	docker container rm -f namenode
	docker container rm -f yarnmaster 
	for ((i=1;i<=10;i++)); 
	do 
		docker container rm -f datanode_$i
	done
	
}

# Create the stockprediction subnetwork
create_network() {
	docker network create --subnet=172.18.0.0/16 stockprediction_network
}

# Destroy the stockprediction subnetwork
destroy_network() {
        docker network rm stockprediction_network
}

# Start the docker service
start_service() {
	sudo systemctl start docker
}

# Terminate the docker service
terminate_service() {
	sudo systemctl stop docker
}

# Build the docker images
build_images() {
	cd base-stockprediction
	docker build -t jbuxofplenty/base-stockprediction .
	cd ..
	cd namenode-stockprediction
	docker build -t jbuxofplenty/namenode-stockprediction .
	cd ..
	cd yarnmaster-stockprediction
	docker build -t jbuxofplenty/yarnmaster-stockprediction .
	cd ..
	cd datanode-stockprediction
	docker build -t jbuxofplenty/datanode-stockprediction .
	cd ..	
}

############################################################
### Perform requested task

if [[ "$TASK" == "INIT" ]] ; then
	init_containers
elif [[ "$TASK" == "KILL" ]] ; then
	kill_containers
elif [[ "$TASK" == "CREATE" ]] ; then
	create_network
elif [[ "$TASK" == "DESTROY" ]] ; then
	destroy_network
elif [[ "$TASK" == "START" ]] ; then
	start_service
elif [[ "$TASK" == "TERMINATE" ]] ; then
	terminate_service
elif [[ "$TASK" == "BUILD" ]] ; then
	build_images
fi
