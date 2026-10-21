#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
: "${MODEL_PATH:?Set MODEL_PATH to the base model directory}"
: "${DATA_DIR:?Set DATA_DIR to the full train/val Parquet directory}"
N_GPUS=${GPUS_PER_NODE:-8}
N_NODES=${NNODES:-1}
export REWARD_CALC_TYPE=answer_line_exact_match_v2
export ANSWER_OVER_FLOW_LIMIT=128
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
export PYTHONPATH="$ROOT/training/verl_4b:${PYTHONPATH:-}"
D=$DATA_DIR
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/checkpoints/longzero_4b}
STEPS=${STEPS:-72}
N_TRAIN=$(python3 -c "import pandas as pd;print(len(pd.read_parquet('$D/train.parquet')))")
EPOCHS=$(( (STEPS + N_TRAIN / 64 - 1) / (N_TRAIN / 64) ))
mkdir -p "$OUTPUT_DIR"
# For multiple nodes, start/join Ray before invoking this command on the head node.
python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$D/train.parquet \
      data.val_files=$D/val.parquet \
      data.train_batch_size=64 \
      data.max_prompt_length=32768 \
      data.max_response_length=8192 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      actor_rollout_ref.model.path="$MODEL_PATH" \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.actor.use_kl_loss=True \
      actor_rollout_ref.actor.kl_loss_coef=0.001 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl \
      actor_rollout_ref.actor.entropy_coeff=0 \
      actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGP_MB:-4} \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
      actor_rollout_ref.rollout.n=8 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOGP_MB:-4} \
      actor_rollout_ref.ref.fsdp_config.param_offload=False \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger='["console"]' \
      trainer.project_name='lczero' \
      trainer.experiment_name="longzero_4b" \
      trainer.n_gpus_per_node="${N_GPUS}" \
      trainer.nnodes="${N_NODES}" \
      trainer.save_freq=18 \
      trainer.test_freq=6 \
      trainer.max_actor_ckpt_to_keep=1 \
      trainer.max_critic_ckpt_to_keep=10 \
      trainer.total_epochs=${EPOCHS} \
      trainer.total_training_steps=${STEPS} \
      trainer.resume_mode=disable \
      actor_rollout_ref.actor.checkpoint.save_contents="['model']" \
      trainer.default_local_dir="$OUTPUT_DIR" \
      "$@"
