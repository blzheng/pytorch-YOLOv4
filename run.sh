while getopts 'L:p:c:s:' OPTION
do
    case $OPTION in
        L)logdir=$OPTARG
            ;;
        p)platform=$OPTARG
            ;;
        c)conda=$OPTARG
            ;;
	s)script=$OPTARG
	    ;;
    esac
done

dataset=${workspace}/pytorch-YOLOv4/coco/images/val2017
precision=("fp32")
if [ "$platform" == "CPX" ]; then
    precision=("bf16")
elif [ "$platform" == "SPR" ]; then
    export DNNL_GRAPH_CONSTANT_CACHE=1
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    precision+=("bf16")
fi

mode=("imperative" "jit")
envs=("stock-pytorch-ipex" "pytorch-ipex" "latest-stock-pytorch-ipex")

for e in ${envs[@]}
do
    $conda activate $e
    export LD_PRELOAD="${conda_path}/$e/lib/libiomp5.so:${conda_path}/$e/lib/libjemalloc.so"
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    ipex=("cpu" "none")
    if [ "$e" == "pytorch-ipex" ]; then
        ipex=("xpu")
    fi
    for datatype in ${precision[@]}
    do
        ARGS="-e $e -L $logdir -p $datatype -d $dataset"
        for m in ${mode[@]}
        do
            M_ARGS="$ARGS -m $m"
            for i in ${ipex[@]} 
            do
                I_ARGS="$M_ARGS -i $i"
                echo "bash $script $I_ARGS"
                bash $script $I_ARGS
                sleep 30
            done
        done
    done
    $conda deactivate 
done
