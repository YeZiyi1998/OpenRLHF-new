ray stop
hostfile=./hostfile_dev
head_node=$(head -1 $hostfile | awk -F " " '{print $1}')

echo $head_node
#ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

tail -n +2 $hostfile | while read line
do
  host=`echo $line | awk -F " " '{print $1}'`
  echo $host
  ssh -n root@$host 'source /data2/rlhf/lixs/l.sh && ray stop'
  #ssh -n root@$host "source /data2/rlhf/lixs/l.sh && ping $head_node"
  #$ssh -n root@$host "ray start --address='$head_node:6379'"
done