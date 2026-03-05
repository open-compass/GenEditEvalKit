#!/usr/bin/env python

init_from=your/download_model/path # Model path downloaded from huggingface (Alpha-VLLM/Lumina-DiMOO)
data_config=configs/data.yaml # You need to define the data in this yaml file.
lr=2e-5
wd=0.1
dropout=0.05
batchsize_per_gpu=2
max_seq_len=5120 # text + image(4096 for 1024 * 1024 resolution) --> token length
exp_name=Lumina-DiMOO-SFT
echo "exp name: $exp_name  node: $SLURMD_NODENAME"
mkdir -p output/"$exp_name"

srun -J Lumina-DiMOO-SFT --partition luminaDLLM --gres gpu:8 --nodes 8 --ntasks-per-node 8 --quotatype reserved \
python -u train/train.py \
--batch_size ${batchsize_per_gpu} \
--accum_iter 4 \
--epochs 2 \
--warmup_epochs 0.001 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 16 \
--output_dir output/"$exp_name" \
--save_iteration_interval 1000 \
--max_seq_len ${max_seq_len} \
--dropout ${dropout} \
--init_from ${init_from} \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"