get -x

   # --use_kl_loss \
   # --use_kl_estimator_k3 \
   # --init_kl_coef 1e-6 \

HDFS_HOME=/apps/OpenRLHF/
RUN_NAME=Qwen2.5_7B_distill_grpo_hybrid
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/apps/openrlhf"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.6 \
   --gamma 1.0 \
   --advantage_estimator group_norm \
   --pretrain deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
   --save_path $HDFS_HOME/checkpoints/$RUN_NAME \
   --ckpt_path $HDFS_HOME/checkpoints/$RUN_NAME \
   --save_hf_ckpt \
   --micro_train_batch_size 1 \
   --train_batch_size 1024 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --num_episodes 20 \
   --prompt_max_len 1024 \
   --max_samples 1000000 \
   --generate_max_len 10240 \
   --temperature 0.6 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-6 \
   --prompt_data pe-nlp/ORZ-Math-57K-R1 \
   --input_key input \
   --label_key ground_truth_answer \
   --save_steps 10 \
   --flash_attn \
   --init_kl_coef 0.0 \
   --load_checkpoint \
   --use_wandb <wandb_api_key> \
   --wandb_run_name $RUN_NAME \
   --wandb_project rl_from_distill \
   --max_ckpt_num 5 \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --vllm_sync_backend nccl \
   --packing_samples \
   --enforce_eager \
   --vllm_enable_sleep \
   --remote_rm_url /apps/OpenRLHF/examples/scripts/r1_reward_func.py