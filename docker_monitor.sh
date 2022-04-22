#!/bin/bash
INTERVAL=0.5 # interval (in seconds for logging information)
OUTNAME=inference_benchmark.txt # log file  name

# function for logging container stats
update_file() {
  docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}},\t{{.MemUsage}},\t{{.NetIO}},\t{{.BlockIO}},\t{{.PIDs}}" | tee --append $OUTNAME;
  echo $(date  +"%Y-%m-%d %T") | tee --append $OUTNAME;
  echo "" | tee --append $OUTNAME
}
while true;
do
    update_file > /dev/null;
    #sleep $INTERVAL;
done