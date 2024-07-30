workdir=$(cd $(dirname $0); pwd)
#hostfile="${workdir}/cirtic_hostfile"
hostfile=$1
#hostfile="${workdir}/actor_hostfile"
#hostfile="${workdir}/reward_hostfile"
#hostfile="${workdir}/inisft_hostfile"
#hostfile="/data2/rlhf/liufeng/d_rlhf/mpi_hostfile"
#hostfile="/data2/rlhf/liufeng/d_rlhf/mpi_hostfile"

cat $hostfile | while read line
do
    host=`echo $line | awk -F " " '{print $1}'`
    echo $host
    #ssh -n yandong@$host 'nvidia-smi | grep bc_torch_2.0 | awk "{print \$5}" | xargs kill -9 '
    ssh -n root@$host 'ps -ef | grep ppo_server | grep -v tsp | awk "{print \$2}" | xargs kill -9 '
    # ssh -n root@$host 'ps -ef | grep node | grep -v tsp | awk "{print \$2}" | xargs kill -9 '
    # ssh -n root@$host 'ps -ef | grep vscode | grep -v tsp | awk "{print \$2}" | xargs kill -9 '
    # ssh -n root@$host 'ps -ef | grep tmux | grep -v tsp | awk "{print \$2}" | xargs kill -9 '
    #ssh -n root@$host 'ps -ef | grep sh | awk "{print \$2}" | xargs kill -9 '
    #ssh -n root@$host 'ps -ef | grep run_all.sh | awk "{print \$2}" | xargs kill -9 '
    #ssh -n root@$host 'mkdir /root/liufeng/tmp'
    # ssh -n root@$host 'rm -rf /root/.cache/torch_extensions/*'
    # ssh -n root@$host 'rm -rf /root/liufeng/tmp/*'
    # ssh -n root@$host 'rm -rf /tmp/*'
    #ssh -n root@$host 'cp /data2/rlhf/liufeng/stop_jops/limits.conf /etc/security/limits.conf'

    #ssh -n root@$host 'export http_proxy=""'
done
