#!/usr/bin/env bash

set -eou pipefail

stage=$1
stop_stage=$2

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Download models and data"
  python prepare_data.py \
    --train_file data/emilia_zh-cosy-tiny-train.jsonl \
    --test_file data/emilia_zh-cosy-tiny-test.jsonl \
    --local_dir data/parquet_tiny

fi


if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: Train the model"
  CUDA_VISIBLE_DEVICES=2 \
  python3 whisper_server.py \
    --port 8001 \
    --model large-v3-turbo
fi


if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: Evaluate the model"
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$HOME/data/llasa-tts-rl-grpo/train.parquet \
      data.val_files=$HOME/data/llasa-tts-rl-grpo/test.parquet \
      data.train_batch_size=8 \
      data.max_prompt_length=512 \
      data.max_response_length=2048 \
      data.truncation='error' \
      actor_rollout_ref.model.path=HKUSTAudio/Llasa-1B \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=8 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.do_sample=true \
      actor_rollout_ref.rollout.temperature=0.8 \
      actor_rollout_ref.rollout.top_p=0.9 \
      actor_rollout_ref.rollout.n=16 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
      custom_reward_function.path=verl/utils/reward_score/tts_cer.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='llasa_tts_grpo' \
      trainer.experiment_name='whisper_cer_reward' \
      trainer.n_gpus_per_node=2 \
      trainer.nnodes=1 \
      trainer.save_freq=128 \
      trainer.test_freq=128 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1 "$@"
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: Test the model"
  export PYTHONPATH=/workspace/CosyVoice
  python3 test_cosyvoice.py \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav
fi