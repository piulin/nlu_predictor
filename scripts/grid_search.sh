#!/usr/bin/env bash

# $1: start_hs
# $2: end_hs
# $3: cuda_device

save_every=2
dropout=0.5
n_epochs=5
model_name="bilstm_adamw"
src_dic=../
batch_size=64
learning_rate=0.001


Wes_Ses=`seq 150 50 450`
Hs=`seq $1 100 $2`


for wes_ses in $Wes_Ses
do

  for hs in $Hs
  do

      itermodelname=${model_name}_wesses${wes_ses}_hs$hs

      python3.9 ${src_dic}main.py \
      --cuda-device $3 \
      -b $batch_size train \
      -C ${src_dic}configs/c1.json \
      -e $n_epochs \
      -o ${src_dic}models/$itermodelname \
      -train-data ${src_dic}../datasets/nlu_traindev/train.json \
      -lr $learning_rate \
      -wes $wes_ses \
      -hs $hs \
      -ses $wes_ses \
      -dropout $dropout \
      --save-every $save_every \
      --no-training-predictions \
      -bidirectional


      model_endings=`seq 0 $save_every $n_epochs`

      for model_ending in $model_endings
      do
        python3.9 ${src_dic}main.py \
        -b $batch_size test \
        -m ${src_dic}models/${itermodelname}.epoch$model_ending  \
        -train-data ${src_dic}../datasets/nlu_traindev/train.json \
        -test-data ${src_dic}../datasets/nlu_traindev/dev.json  \
        -O ${src_dic}preds/${itermodelname}.epoch$model_ending.json


      done


  done

done
