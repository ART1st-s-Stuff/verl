set -x

if [ "$#" -lt 3 ]; then
    echo "Usage: run_navigation_latent_sft.sh <nproc_per_node> <train_parquet> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
train_parquet=$2
save_path=$3
shift 3

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_parquet \
    data.val_files=null \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length=2048 \
    data.micro_batch_size_per_gpu=1 \
    data.latent_sft.enable=true \
    data.latent_sft.hybrid_fill.enable=true \
    data.latent_sft.action_start_token="<|action_start|>" \
    model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    trainer.default_local_dir=$save_path \
    trainer.project_name=navigation-latent-sft \
    trainer.experiment_name=navigation-latent-sft-qwen-0.5b \
    trainer.logger=console \
    trainer.total_training_steps=1000 \
    $@
