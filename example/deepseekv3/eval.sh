#!/bin/bash


mf_path=$1
worker_num=$2

base_path=$(cd "$(dirname $0)"; pwd)
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export GSLOG=1
export PYTHONPATH=${mf_path}:$PYTHONPATH

msrun --worker_num=${worker_num} --local_worker_num=${worker_num} --master_port=8111 --log_dir=log_infer_dsv3 --join=True --cluster_time_out=300 python ${base_path}/eval_ceval.py -c ${mf_path}/research/deepseek3/deepseek3_671b/predict_deepseek3_671B.yaml -s ${base_path}/../../tests/data/ceval-dataset/dev -t ceval > infer_dsv3_log 2>&1 &
tail -f log_infer_dsv3/worker_0.log
