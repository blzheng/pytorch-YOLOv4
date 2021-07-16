while getopts 'L:p:c:' OPTION
do
    case $OPTION in
        L)logdir=$OPTARG
            ;;
        p)platform=$OPTARG
            ;;
        c)conda=$OPTARG
            ;;
    esac
done

dataset=""
precision=("fp32")
if [ "$platform" == "CPX" ]; then
    precision=("bf16")
elif [ "$platform" == "SPR" ]; then
    export DNNL_GRAPH_CONSTANT_CACHE=1
    export DNNL_MAX_CPU_ISA=AVX512_CORE_AMX
    export LD_PRELOAD="/home/sdp/bzheng/workspace/gperftools-2.7.90/.libs/libtcmalloc.so"
    dataset=/home/sdp/bzheng/workspace/pytorch-YOLOv4/coco/images/val2017
    precision+=("bf16")
elif [ "$platform" == "ICX" ]; then
    export LD_PRELOAD="/home/sdp/bzheng/workspace/gperftools-2.7.90/.libs/libtcmalloc.so"
    dataset=/home/sdp/bzheng/workspace/pytorch-YOLOv4/coco/images/val2017
fi

mode=("imperative" "jit")
envs=("stock-pytorch-ipex" "pytorch-ipex" "latest-stock-pytorch-ipex")


for e in ${envs[@]}
do
    $conda activate $e
    ipex=("cpu" "none")
    if [ "$e" == "pytorch-ipex" ]; then
        ipex=("xpu")
    fi
    for datatype in ${precision[@]}
    do
        ARGS="-e $e -L $logdir -p $datatype -d $dataset"
        for m in ${mode[@]}
        do
            ARGS="$ARGS -m $m"
            for i in ${ipex[@]} 
            do
                ARGS="$ARGS -i $i"
                echo "bash run_multi_instance.sh $ARGS"
                bash run_multi_instance.sh $ARGS
                sleep 30
            done
        done
    done
    $conda deactivate $e
done
