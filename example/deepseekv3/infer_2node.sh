#!/bin/bash
export GSLOG=1
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=AddRmsNormDynamicQuant,PagedAttention
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export MS_ALLOC_CONF="enable_vmm:False"
export MS_PARALLEL_DISPATCH_NUM=4 #2
export MS_ENABLE_SYNC_COPY_INPUT=1

mf_path=$1
quant_type=$2
node_id=$3
cur_ip=$4
master_ip=$5

export HCCL_IF_IP=${cur_ip}
export PYTHONPATH=${mf_path}:${PYTHONPATH}

bash ${mf_path}/scripts/msrun_launcher.sh "quant_infer.py --config ${mf_path}/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml --approach ${quant_type}" 16 8 ${master_ip} 8432 ${node_id} output/msrun_log False 300