# !/bin/bash
export GSLOG=1
export MS_ENABLE_LCCL=off
export HCCL_OP_EXPANSION_MODE=AIV
export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST=AddRmsNormDynamicQuant,PagedAttention
export MS_DEV_RUNTIME_CONF="parallel_dispatch_kernel:True"
export HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT=TRUE
export MS_ALLOC_CONF="enable_vmm:False"
export MS_PARALLEL_DISPATCH_NUM=4 #2
export MS_ENABLE_SYNC_COPY_INPUT=1
base_path=$(cd "$(dirname $0)"; pwd)

node_id=$1
cur_ip=$2
master_ip=$3
mf_path=$4

export HCCL_IF_IP=${cur_ip}
export PYTHONPATH=${mf_path}:${PYTHONPATH}

bash ${mf_path}/scripts/msrun_launcher.sh "ds_infer_a16w4.py --yaml_file ${mf_path}/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b.yaml" 16 8 ${master_ip} 8432 ${node_id} output/msrun_log False 300