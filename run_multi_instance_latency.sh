export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

while getopts 'L:p:d:m:e:i:' OPTION
do
    case $OPTION in
        L)logdir=$OPTARG
            ;;
        p)precision=$OPTARG
            ;;
        d)dataset=$OPTARG
            ;;
        m)mode=$OPTARG
            ;;
        i)ipex=$OPTARG
            ;;
        e)env=$OPTARG
            ;;
    esac
done

if [ ! -d $logdir ]; then
    mkdir -p $logdir
fi

ARGS=""
log_name=$logdir
if [ "$precision" == "bf16" ]; then
    ARGS="$ARGS --bf16"
fi
if [ "$mode" == "jit" ]; then
    ARGS="$ARGS --jit"
fi
if [ "$ipex" == "cpu" ]; then
    ARGS="$ARGS --ipex"
elif [ "$ipex" == "xpu" ]; then
    ARGS="$ARGS --ipex --xpu"
elif [ "$ipex" == "none" ]; then
    if [ "$env" == "stock-pytorch-ipex" ]; then
	env="stock-pytorch"
    elif [ "$env" == "latest-stock-pytorch-ipex" ]; then
	env="latest-stock-pytorch"
    fi
fi

echo "args: $ARGS"

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
CORES_PER_INSTANCE=4
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
BATCH_SIZE=1

log_name="${logdir}/latency_bs${BATCH_SIZE}_${precision}_${env}_${mode}.log"

export OMP_NUM_THREADS=$CORES_PER_INSTANCE
export $KMP_SETTING
echo -e "### using OMP_NUM_THREADS=$CORES_PER_INSTANCE"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

INSTANCES=`expr $TOTAL_CORES / $CORES_PER_INSTANCE`
LAST_INSTANCE=`expr $INSTANCES - 1`
INSTANCES_PER_SOCKET=`expr $INSTANCES / $SOCKETS`
for i in $(seq 1 $LAST_INSTANCE); do
    numa_node_i=`expr $i / $INSTANCES_PER_SOCKET`
    start_core_i=`expr $i \* $CORES_PER_INSTANCE`
    end_core_i=`expr $start_core_i + $CORES_PER_INSTANCE - 1`
    
    echo "### running on instance $i, numa node $numa_node_i, core list {$start_core_i, $end_core_i}..."
    numactl --physcpubind=$start_core_i-$end_core_i --membind=$numa_node_i python -u main.py --evaluate $ARGS -dir $dataset -b ${BATCH_SIZE} -j 0 2>&1 | tee ${log_name}${i} &
done

numa_node_0=0
start_core_0=0
end_core_0=`expr $CORES_PER_INSTANCE - 1`

echo "### running on instance 0, numa node $numa_node_0, core list {$start_core_0, $end_core_0}...\n\n"
numactl --physcpubind=$start_core_0-$end_core_0 --membind=$numa_node_0 python -u main.py --evaluate $ARGS -dir $dataset -b ${BATCH_SIZE} -j 0 2>&1 | tee ${log_name}0
