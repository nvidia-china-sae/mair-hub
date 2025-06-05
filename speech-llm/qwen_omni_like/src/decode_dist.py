# Copyright (c) 2024 Tsinghua Univ. (authors: Xingchen Song)
#               2025               （authors: Yuekai Zhang）
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from https://github.com/xingchensong/S3Tokenizer/blob/main/s3tokenizer/cli.py
""" Example Usage
split=test_zh
llm_path=f5-tts/exp_zh/checkpoint-805000
huggingface-cli download --local-dir f5-tts-small-wenetspeech4tts-basic yuekai/f5-tts-semantic-token-small-wenetspeech4tts-basic
model_path=f5-tts-small-wenetspeech4tts-basic/epoch-10-avg-5.pt
huggingface-cli download nvidia/bigvgan_v2_24khz_100band_256x --local-dir ./bigvgan_v2_24khz_100band_256x
vocoder=./bigvgan_v2_24khz_100band_256x
torchrun --nproc_per_node=2 \
    f5-tts/infer_dist.py \
                --output_dir $output_dir \
                --batch_size 1 \
                --num_workers 2 \
                --llm-model-name-or-path $llm_path \
                --flow-matching-model-path $model_path \
                --decoder-dim 768 --nhead 12 --num-decoder-layers 18 \
                --use-cosyvoice-semantic-token True \
                --vocoder-dir $vocoder \
                --split-name $split -top-k 50 -top-p 0.95 -temperature 0.8 \
                --tokenizer-dir Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import whisper
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from train import DEFAULT_SPEECH_TOKEN, add_model_arguments, add_training_arguments, get_model
from transformers import AutoTokenizer
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

# https://github.com/FunAudioLLM/CosyVoice/tree/main/third_party
# sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
try:
    torch.multiprocessing.set_start_method("spawn")
except RuntimeError:
    pass


def get_args():
    parser = argparse.ArgumentParser(description="extract speech code")
    parser.add_argument(
        "--split-name",
        type=str,
        default="test",
        help="huggingface dataset split name",
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="commoneval",
        help="subset name",
    )
    parser.add_argument(
        "--output-dir", required=True, type=str, help="dir to save result"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="batch size (per-device) for inference",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="workers for dataloader"
    )
    parser.add_argument(
        "--prefetch", type=int, default=2, help="prefetch for dataloader"
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="Romeo and Juliet might be the most famous act of William Shakespeare.",
        help="The prompt text",
    )

    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default="./assets/common_voice_en_2586258.wav",
        help="The path to the prompt speech",
    )
    add_model_arguments(parser)
    add_training_arguments(parser)
    args = parser.parse_args()
    return args


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    print(
        "Inference on multiple gpus, this gpu {}".format(local_rank)
        + ", rank {}, world_size {}".format(rank, world_size)
    )
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return world_size, local_rank, rank


def preprocess(
    messages,
    tokenizer,
):
    """Preprocesses the data for supervised fine-tuning."""
    texts = []
    TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                tokenize=True,
                add_generation_prompt=False,
                chat_template=TEMPLATE,
                padding="longest",
                truncation=False,
            )
        )
    max_len_texts = max([len(text) for text in texts])
    if tokenizer.padding_side == "right":
        texts = [
            text + [tokenizer.pad_token_id] * (max_len_texts - len(text))
            for text in texts
        ]
    else:
        texts = [
            [tokenizer.pad_token_id] * (max_len_texts - len(text)) + text
            for text in texts
        ]

    input_ids = torch.tensor(texts, dtype=torch.int)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return input_ids, attention_mask


def custom_collate(batch):
    assert len(batch) == 1
    audio = batch[0]["audio"]
    assert audio["sampling_rate"] == 16000
    result = {"audio": audio["array"]}
    for keys in batch[0].keys():
        if keys != "audio":
            result[keys] = batch[0][keys]
    return result


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert torch.cuda.is_available()
    world_size, local_rank, rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dataset = load_dataset(
        "hlt-lab/voicebench",
        args.subset_name,
        split=args.split_name,
        trust_remote_code=True,
    )

    model, tokenizer = get_model(args)
    model.to(device)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch,
        collate_fn=custom_collate,
    )

    total_steps = len(dataset)

    if rank == 0:
        progress_bar = tqdm(total=total_steps, desc="Processing", unit="wavs")

    message = [
        {"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}"},
        {"role": "assistant", "content": ""},
    ]
    input_ids, attention_mask = preprocess([message], tokenizer)
    results_jsonl_file = open(
        os.path.join(
            args.output_dir,
            f"results-{args.subset_name}-{args.split_name}-{rank}-audio.jsonl",
        ),
        "w",
    )

    if args.enable_speech_output:
        from cosyvoice.cli.cosyvoice import CosyVoice2
        from cosyvoice.utils.file_utils import load_wav
        import sys
        sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")
        from web_demo import audio_decode_cosyvoice2
        token2wav_model = CosyVoice2(
            model_path=args.token2wav_path,
            device=device,
            use_lora=args.use_lora,
        )
        prompt_speech_16k = load_wav(args.prompt_speech_path, 16000)

    for batch_idx, batch in enumerate(dataloader):
        audio = batch["audio"]
        audio = torch.from_numpy(audio).to(device).to(torch.float32)
        fbank = whisper.log_mel_spectrogram(audio, device=device)
        fbank = fbank.unsqueeze(0)
        if args.enable_speech_output:
            generated_ids, audio_tokens = model.decode_with_speech_output(
                fbank, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
            )
            audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)
            audio_hat = audio_decode_cosyvoice2(
                audio_tokens,
                args.prompt_text,
                prompt_speech_16k,
                token2wav_model,
            )
            speech_file_name = os.path.join(args.output_dir, f"rank{rank}_idx_{batch_idx}.wav")
            sf.write(speech_file_name, audio_hat.squeeze(0).cpu().numpy(), 24000)
        else:
            generated_ids = model.decode(
                fbank, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
            )
        hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        result_dict = {}
        for key in batch.keys():
            if key != "audio":
                result_dict[key] = batch[key]
        result_dict["response"] = hyps[0]
        if args.enable_speech_output:
            result_dict["output_wav_path"] = speech_file_name
        if rank == 0:
            print(result_dict)
        json.dump(result_dict, results_jsonl_file)
        results_jsonl_file.write("\n")

        if rank == 0:
            progress_bar.update(world_size * args.batch_size)

    if rank == 0:
        progress_bar.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
