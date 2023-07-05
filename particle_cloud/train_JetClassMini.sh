#!/bin/bash

set -x

echo "args: $@"

# set the dataset dir via `DATADIR`
[[ -z $DATADIR ]] && DATADIR='../JetClassMini'

# set a comment via `COMMENT`
suffix=${COMMENT}

epochs=20
dataopts="--num-workers 0 --fetch-step 1"

# model: ParT, PN, PFN, PCNN
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 128 --start-lr 2e-4"
elif [[ "$model" == "ParT-lite" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp -o num_layers 3 -o num_cls_layers 1"
    batchopts="--batch-size 128 --start-lr 2e-4"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 128 --start-lr 2e-3"
elif [[ "$model" == "PN-lite" ]]; then
    modelopts="networks/example_ParticleNet.py -o conv_params [(7,(32,32,32)),(7,(64,64,64))]"
    batchopts="--batch-size 128 --start-lr 2e-3"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 1024 --start-lr 5e-3"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 1024 --start-lr 5e-3"
else
    echo "Invalid model $model!"
    exit 1
fi

weaver \
    --data-train \
    "TTBar:${DATADIR}/TTBar_*.root" \
    "ZJetsToNuNu:${DATADIR}/ZJetsToNuNu_*.root" \
    --data-test \
    "TTBar:${DATADIR}/TTBar_*.root" \
    "ZJetsToNuNu:${DATADIR}/ZJetsToNuNu_*.root" \
    --data-config data/JetClassMini.yaml --network-config $modelopts \
    --model-prefix training/JetClassMini/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --num-epochs 20 --gpus '' \
    --optimizer ranger --log logs/JetClassMini_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClassMini_${model}${suffix} \
    "${@:2}"
