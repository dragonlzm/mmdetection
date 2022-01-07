#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
DATA_DIR=$4

PORT=${PORT:-29500}

rm -rf ./data
ln -sf ${DATA_DIR} ./data

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:5}
