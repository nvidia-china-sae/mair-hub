#!/usr/bin/env bash
export PYTHONPATH=$PYTHONPATH:/workspace/CosyVoice
set -eou pipefail

stage=$1
stop_stage=$2

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}


ngpu=4
exp_dir=./exp/exp_ultrachat_voiceassistant_sft

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "stage 0: Clone CosyVoice repo and install requirements inside the container"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /workspace/CosyVoice
  cd /workspace/CosyVoice && git submodule update --init --recursive && cd -
  pip install -r requirements.txt
  pip install -r requirements-cosyvoice.txt

  huggingface-cli download --local-dir models/CosyVoice2-0.5B FunAudioLLM/CosyVoice2-0.5B

  # For Gradio demo, we follow https://arxiv.org/abs/2412.15649 to use ASR model to decode the history speech as context.
  pip install sherpa-onnx
  model_path=models/sherpa-onnx-paraformer-zh-2023-09-14
  if [ ! -d $model_path ]; then
    wget -nc https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2 -C models
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "stage 1: Gradio Demo"
  python3 ./src/web_demo.py \
    --speech-encoder-path-or-name models/large-v2.pt  \
    --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
    --checkpoint-path $exp_dir/epoch-1/pytorch_model.bin \
    --use-flash-attn True \
    --enable-speech-output True \
    --use-lora True --token2wav-path models/CosyVoice2-0.5B --share
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "stage 2: Server for voicebench speech2text evaluation"
  N_GPUS=2 # lauch multiple servers to handle multiple clients, since only batch_size=1 is supported for now
  for id in $(seq 0 $(($N_GPUS - 1)))
  do
    log "Launching server on GPU $id with port $(expr 8000 + $id)"
    CUDA_VISIBLE_DEVICES=$id python3 ./src/server.py \
      --speech-encoder-path-or-name models/large-v2.pt  \
      --llm-path-or-name models/Qwen2.5-0.5B-Instruct \
      --checkpoint-path $exp_dir/checkpoint-10/pytorch_model.bin \
      --use-flash-attn True \
      --enable-speech-output False \
      --port $(expr 8000 + $id) \
      --use-lora True &
  done
  wait
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "stage 3: VoiceBench Client"
  # see https://github.com/MatthewCYM/VoiceBench
  # declare -a target_datasets=("alpacaeval_full" "wildvoice" "ifeval" "commoneval" "openbookqa" "sd-qa" "advbench" "bbh" "mmsu")
  declare -a target_datasets=("commoneval")

  NUM_CLIENT_JOBS=$ngpu
  BASE_PORT=8000

  log "Starting $NUM_CLIENT_JOBS parallel client jobs to process ${#target_datasets[@]} datasets."

  for job_id in $(seq 0 $(($NUM_CLIENT_JOBS - 1)))
  do
    ( # Start a subshell for backgrounding this client job's tasks
      current_port=$(expr $BASE_PORT + $job_id)
      log "Client Job $job_id: Initializing. Will connect to port $current_port."
      
      processed_count_for_this_job=0
      # Iterate over all datasets using their indices
      for i in "${!target_datasets[@]}"; do
        # Assign dataset to job_id in a round-robin fashion
        if [ $(($i % $NUM_CLIENT_JOBS)) -eq $job_id ]; then
          dataset="${target_datasets[$i]}"
          
          # local split_name # Determine split_name based on dataset
          if [ "$dataset" == "sd-qa" ]; then
            split_name="usa"
          else
            split_name="test"
          fi
          
          log "Client Job $job_id (Port $current_port): Processing dataset '$dataset' (split '$split_name')"
          python3 ./src/client.py \
            --subset-name "$dataset" \
            --split-name "$split_name" \
            --output-dir "$exp_dir/results" \
            --port "$current_port"
          
          if [ $? -ne 0 ]; then
            log "Client Job $job_id (Port $current_port): ERROR processing dataset '$dataset'."
          fi
          processed_count_for_this_job=$(($processed_count_for_this_job + 1))
        fi
      done
      log "Client Job $job_id (Port $current_port): Finished. Processed $processed_count_for_this_job datasets."
    ) & # Run this client job's subshell in the background
  done

  log "All client jobs launched. Waiting for completion..."
  wait
  log "All client jobs have completed."
fi