#!/usr/bin/env bash

set -eou pipefail

stage=$1
stop_stage=$2

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


export PYTHONPATH=/workspace/CosyVoice

if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  log "stage -1: install vllm and CosyVoice"
  git clone https://github.com/vllm-project/vllm.git
  cd vllm
  USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
  pip install --no-deps -e .

  git clone https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
  pip install -r ./requirements-cosyvoice.txt

  modelscope download --model iic/CosyVoice2-0.5B --local-dir /workspace/CosyVoice2-0.5B
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: prepare data into verl format"
  python prepare_data.py \
    --train_file data/emilia_zh-cosy-tiny-train.jsonl \
    --test_file data/emilia_zh-cosy-tiny-test.jsonl \
    --local_dir data/parquet_tiny
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: start token2wav asr server for reward function"
  python3 token2wav_asr_server.py

  # log "Test the reward server"
  # python3 reward_tts.py \
  #   --input data/emilia_zh-cosy-tiny-test.jsonl \
  #   --no-interactive --debug
fi 

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: grpo train"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export MKL_SERVICE_FORCE_INTEL=TRUE
  n_gpus_per_node=8
  micro_batch_size=4
  train_batch_size=32
  python3 -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=data/parquet_aishell3/train.parquet \
      data.val_files=data/parquet_aishell3/test.parquet \
      data.train_batch_size=$train_batch_size \
      data.max_prompt_length=1024 \
      data.max_response_length=1024 \
      data.truncation='error' \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.path='/workspace/rl/llasa_cosyvoice2_token_qwen_0.5b/checkpoint-885000' \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
      reward_model.reward_manager=prime \
      custom_reward_function.path=reward_tts.py \
      custom_reward_function.name=compute_score \
      trainer.project_name='tts_grpo' \
      trainer.experiment_name='aishell3_reward_tts_prime' \
      trainer.logger=['console','wandb'] \
      trainer.n_gpus_per_node=$n_gpus_per_node \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=400 \
      trainer.resume_mode='auto' \
      trainer.total_epochs=1
fi

step=3000
llm_path=./checkpoints/tts_grpo/aishell3_reward_tts_prime/global_step_${step}
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: merge the model"
  python -m verl.model_merger merge \
      --backend fsdp \
      --local_dir $llm_path/actor \
      --target_dir $llm_path/merged_hf_model || exit 1
    
fi 
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "stage 4: Test the model"
  dataset=zero_shot_zh
  output_dir=./outputs_rl_aishell3_step${step}_${dataset}_jit_trt_fp16_reward_tts

  token2wav_path=/workspace/CosyVoice2-0.5B
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 \
      infer_dist.py \
                  --output-dir $output_dir \
                  --llm-model-name-or-path $llm_path/merged_hf_model \
                  --token2wav-path $token2wav_path \
                  --split-name ${dataset} || exit 1

  bash scripts/compute_wer.sh $output_dir ${dataset}
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "stage 5: Infer with single case"
  python3 infer.py \
    --token2wav-path /workspace/CosyVoice2-0.5B \
    --prompt-text "吃燕窝就选燕之屋，本节目由26年专注高品质燕窝的燕之屋冠名播出。豆奶牛奶换着喝，营养更均衡，本节目由豆本豆豆奶特约播出。" \
    --prompt-speech-path ./assets/prompt_audio.wav \
    --model-path $llm_path/merged_hf_model \
    --input-text "扁担长，板凳宽，扁担绑在板凳上。吃葡萄不吐葡萄皮，不吃葡萄倒吐葡萄皮。"
fi
