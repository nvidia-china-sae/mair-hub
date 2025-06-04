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
  log "stage 0: Clone CosyVoice repo and install requirements inside the container"
  # docker: ghcr.io/swivid/f5-tts:main
  pip install -r qwen_omni/requirements.txt

  wget https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt -O models/large-v2.pt
  huggingface-cli download --local-dir models/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-0.5B-Instruct
  
  huggingface-cli download --local-dir data/librispeech_asr --repo-type dataset fixie-ai/librispeech_asr 
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: Training Speech2Speech model's adaptor only"
  exp_dir=./exp/exp_librispeech_adapter_pretrain
  ngpu=4

  train_cmd_args="--max-duration 800 --on-the-fly-feats True \
  --exp-dir $exp_dir  \
  --huggingface-dataset-path-or-name data --dataset librispeech \
  --speech-encoder-path-or-name models/large-v2.pt \
  --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
  --deepspeed \
  --deepspeed_config ./src/ds_config_zero1.json \
  --use-flash-attn True \
  --use-lora False --unfreeze-llm False --unfreeze-speech-projector True --enable-speech-output False"

  latest_checkpoint_step=-1
  if [ -d "$exp_dir" ]; then
    # List directories matching checkpoint-* and find the one with the largest step number
    for checkpoint_dir in $(ls -d $exp_dir/checkpoint-*/ 2>/dev/null | sort -V); do
      checkpoint_name=$(basename "$checkpoint_dir") # e.g., checkpoint-1000
      # Extract step number using parameter expansion
      current_step=${checkpoint_name#checkpoint-}
      # Ensure current_step is a number
      if [[ "$current_step" =~ ^[0-9]+$ ]] && [ "$current_step" -gt "$latest_checkpoint_step" ]; then
        latest_checkpoint_step=$current_step
      fi
    done
  fi
  if [ "$latest_checkpoint_step" -ge 0 ]; then
    log "Continuing training from checkpoint-$latest_checkpoint_step"
    step=$latest_checkpoint_step
    train_cmd_args="$train_cmd_args --pretrained-model-path $exp_dir/checkpoint-${step}/pytorch_model.bin --sampler-state-dict-path $exp_dir/checkpoint-${step}/sampler.pt"
  else
    log "Starting training from scratch as no checkpoint was found in $exp_dir"
    # No pretrained model or sampler state dict needed for the first run
  fi

  torchrun --nproc_per_node $ngpu --nnodes $SLURM_JOB_NUM_NODES --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT --rdzv_backend c10d --rdzv_id $SLURM_JOBID ./src/train.py \
    $train_cmd_args
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: Training Speech2Speech Model"
  ngpu=4
  exp_dir=./exp/exp_ultrachat_voiceassistant_sft
  
  torchrun --nproc_per_node $ngpu ./qwen_omni/train.py \
    --max-duration 150 \
    --enable-musan False \
    --exp-dir $exp_dir \
    --speech-encoder-path-or-name models/large-v2.pt \
    --llm-path-or-name Qwen/Qwen2.5-0.5B-Instruct \
    --dataset-format vocalnet \
    --manifest-dir data/fbank \
    --deepspeed \
    --deepspeed_config ./qwen_omni/ds_config_zero1.json \
    --use-flash-attn True --on-the-fly-feats True \
    --use-lora True --unfreeze-llm True --unfreeze-speech-projector True --enable-speech-output True
fi

