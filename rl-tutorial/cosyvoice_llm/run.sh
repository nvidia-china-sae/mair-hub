#!/usr/bin/env bash

set -eou pipefail

stage=$1
stop_stage=$2

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: download whisper model"
  wget https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt

  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
  pip install --no-deps -e .
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: prepare data into verl format"
  # python prepare_data.py \
  #   --train_file data/emilia_zh-cosy-tiny-train.jsonl \
  #   --test_file data/emilia_zh-cosy-tiny-test.jsonl \
  #   --local_dir data/parquet_tiny
  # python prepare_data.py \
  #   --train_file data/aishell-3-cosy.jsonl \
  #   --test_file data/emilia_test.jsonl \
  #   --local_dir data/parquet_aishell3
  python prepare_data.py \
    --train_file data/aishell3_small.jsonl \
    --test_file data/emilia_small.jsonl \
    --local_dir data/parquet_aishell3_small
fi

export PYTHONPATH=/workspace/CosyVoice
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: start whisper server"
  # wget https://openaipublic.azureedge.net/main/whisper/models/aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a/large-v3-turbo.pt
  CUDA_VISIBLE_DEVICES=7 \
  python3 whisper_server.py \
    --port 8001 \
    --model models/large-v3-turbo.pt \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav

  # python3 whisper_server.py \
  #   --port 8001 \
  #   --model models/distill-whisper-large-v2-multi-hans-epoch-6-avg-8.pt \
  #   --token2wav-path /workspace/CosyVoice2-0.5B \
  #   --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
  #   --prompt-speech-path ./assets/prompt_audio.wav \
  #   --remove-whisper-encoder-input-length-restriction
fi

# export PYTHONPATH=/
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: Evaluate the model"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  n_gpus_per_node=7
  micro_batch_size=4
  train_batch_size=28
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=data/parquet_aishell3/train.parquet \
      data.val_files=data/parquet_aishell3/test.parquet \
      data.train_batch_size=$train_batch_size \
      data.max_prompt_length=1024 \
      data.max_response_length=1024 \
      data.truncation='error' \
      actor_rollout_ref.model.path='/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000' \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=14 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.actor.use_kl_loss=False \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      actor_rollout_ref.actor.fsdp_config.param_offload=False \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
      actor_rollout_ref.rollout.do_sample=true \
      actor_rollout_ref.rollout.temperature=0.8 \
      actor_rollout_ref.rollout.top_p=0.9 \
      actor_rollout_ref.rollout.n=4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
      custom_reward_function.path=tts_cer.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='llasa_tts_grpo' \
      trainer.experiment_name='whisper_turbo_per_aishell3' \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=400 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1
fi


if [ $stage -le 20 ] && [ $stop_stage -ge 20 ]; then
  log "stage 20: Evaluate the model"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  n_gpus_per_node=2
  micro_batch_size=1
  train_batch_size=$((n_gpus_per_node * micro_batch_size))
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=data/parquet_aishell3_small/train.parquet \
      data.val_files=data/parquet_aishell3_small/test.parquet \
      data.train_batch_size=$train_batch_size \
      data.max_prompt_length=512 \
      data.max_response_length=2048 \
      data.truncation='error' \
      actor_rollout_ref.model.path='/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000' \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=$train_batch_size \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size \
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
      actor_rollout_ref.rollout.n=2 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=true \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
      actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
      custom_reward_function.path=tts_cer.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='llasa_tts_grpo' \
      trainer.experiment_name='whisper_cer_reward_test_aishell3' \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      trainer.nnodes=1 \
      trainer.save_freq=128 \
      trainer.test_freq=128 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: Test the model"
  python3 test_cosyvoice.py \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "stage 4: Test the model"
  python3 tts_cer.py \
    --input data/emilia_zh-cosy-tiny-test.jsonl \
    --max-samples 5
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "stage 5: Test the model"
  dataset=test_zh
  output_dir=./outputs_rl_aishell3_step100_${dataset}
  # llm_path=/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000
  # llm_path=/workspace/slam/mair-hub/rl-tutorial/cosyvoice_llm/checkpoints/llasa_tts_grpo/whisper_cer_reward_tiny/global_step_50/merged_hf_model
  llm_path=/workspace/slam/mair-hub/rl-tutorial/cosyvoice_llm/checkpoints/llasa_tts_grpo/whisper_turbo_per_aishell3/global_step_500/merged_hf_model
  token2wav_path=/workspace/CosyVoice2-0.5B

  torchrun --nproc_per_node=8 \
      infer_dist.py \
                  --output-dir $output_dir \
                  --llm-model-name-or-path $llm_path \
                  --token2wav-path $token2wav_path \
                  --split-name ${dataset} || exit 1
                  # --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
                  # --prompt-speech-path ./assets/prompt_audio.wav || exit 1
  bash scripts/compute_wer.sh $output_dir ${dataset}
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "stage 6: merge the model"
  llm_path=/workspace/slam/mair-hub/rl-tutorial/cosyvoice_llm/checkpoints/llasa_tts_grpo/whisper_cer_reward_tiny/global_step_50
  llm_path=/workspace/slam/mair-hub/rl-tutorial/cosyvoice_llm/checkpoints/llasa_tts_grpo/whisper_turbo_per_aishell3/global_step_500
  python /workspace/rl/ch-tts-llasa-rl-grpo/scripts/model_merger.py merge \
      --backend fsdp \
      --local_dir $llm_path/actor \
      --target_dir $llm_path/merged_hf_model

  # python -m verl.model_merger merge \
  #     --backend fsdp \
  #     --local_dir $llm_path/actor \
  #     --target_dir $llm_path/merged_hf_model
    
fi 