#/bin/bash

# CIL CONFIG
NOTE="l2p" # Short description of the experiment. (WARNING: logs/results with the same note will be overwritten!)
MODE="l2p"
DATASET="cifar100" # cifar10, cifar100, tinyimagenet, imagenet
N_TASKS=5
N=50
M=10
GPU_TRANSFORM=False
USE_AMP="--use_amp"
SEEDS=1

if [ "$DATASET" == "cifar10" ]; then
    MEM_SIZE=500 ONLINE_ITER=1 TOTAL_CLASS=10
    MODEL_NAME="resnet18" EVAL_PERIOD=100
    BATCHSIZE=16; LR=0.01 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "cifar100" ]; then
    MEM_SIZE=2000 ONLINE_ITER=3 TOTAL_CLASS=100
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=16; LR=0.01 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "tinyimagenet" ]; then
    MEM_SIZE=4000 ONLINE_ITER=3
    MODEL_NAME="resnet34" EVAL_PERIOD=100
    BATCHSIZE=16; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=1

elif [ "$DATASET" == "imagenet" ]; then
    N_TASKS=10 MEM_SIZE=20000 ONLINE_ITER=0.25
    MODEL_NAME="resnet34" EVAL_PERIOD=1000
    BATCHSIZE=256; LR=3e-4 OPT_NAME="adam" SCHED_NAME="default" IMP_UPDATE_PERIOD=10

else
    echo "Undefined setting"
    exit 1
fi


CUDA_VISIBLE_DEVICES=1 python main.py --mode $MODE \
--dataset $DATASET \
--num_tasks $N_TASKS --m $M --n $N \
--seed $SEEDS \
--model_name $MODEL_NAME --optim $OPT_NAME --sched_name $SCHED_NAME \
--lr $LR --batchsize $BATCHSIZE \
--memory_size $MEM_SIZE --gpu_transform $GPU_TRANSFORM --online_iter $ONLINE_ITER \
--note $NOTE --eval_period $EVAL_PERIOD --imp_update_period $IMP_UPDATE_PERIOD --total_class $TOTAL_CLASS

