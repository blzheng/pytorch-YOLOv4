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

log_name="${logdir}/accuracy_${precision}_${env}_${mode}.log"

if [ ! -d predictions/${precision}_${env}_${mode}/ ]; then
    mkdir -p predictions/${precision}_${env}_${mode}/
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
TOTAL_CORES=$CORES
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
BATCH_SIZE=128

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
echo -e "### using $KMP_SETTING\n\n"
sleep 3

start_core=0
end_core=`expr $CORES - 1`

numactl --physcpubind=$start_core-$end_core --membind=0 python -u main.py --evaluate $ARGS -dir $dataset -b $BATCH_SIZE --max_iter -1 2>&1 | tee ${log_name}
# numactl --physcpubind=$start_core-$end_core --membind=0 python -u main.py --evaluate $ARGS -dir $dataset -b $BATCH_SIZE --max_iter 10 --draw --draw_dir predictions/${precision}_${env}_${mode}/ 2>&1 | tee ${log_name}
