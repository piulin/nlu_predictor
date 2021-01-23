#!/usr/bin/env bash

step=5
start=120
iters=5
model_name=bilstm
current_iter=$start
src_dic=../

for i in `seq 1 $iters`
do

  next_iter=$[ $current_iter+$step ]
  python3.9 ${src_dic}main.py \
  -b 64 continue \
  -e $step \
  -m ${src_dic}models/$model_name$current_iter \
  -train-data ${src_dic}../datasets/nlu_traindev/train.json \
  -lr 0.001 \
  -o ${src_dic}models/$model_name$next_iter

  python3.9 ${src_dic}main.py \
  -b 64 test \
  -m ${src_dic}models/$model_name$next_iter  \
  -train-data ${src_dic}../datasets/nlu_traindev/train.json \
  -test-data ${src_dic}../datasets/nlu_traindev/dev.json  \
  -O ${src_dic}preds/$model_name$next_iter.json

  current_iter=$next_iter

done

