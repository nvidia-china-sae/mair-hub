"""Lightweight REST server that returns Whisper NLL (and transcript).

Usage
-----
```bash
# GPU 3 only
CUDA_VISIBLE_DEVICES=3 python whisper_server.py --port 8000 --model large-v3 --token2wav-path /path/to/token2wav
```

Client (reward function) can POST JSON:
```json
{
  "tokens": [123, 456, ...],
  "text": "안녕하세요"
}
```
Response:
```json
{
  "nll": 4.7321,
  "transcript": "안녕하세요"
}
```

You may also POST raw *wav* bytes to `/score_wav` if you want to keep
decoding client‑side.
"""

from __future__ import annotations

import argparse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import sys

import torch
import whisper  # type: ignore
import whisper.audio as _wa
from torch.nn.functional import cross_entropy
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")

# ---------------------------------------------------------------------------
# CLI / model loading
# ---------------------------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Whisper NLL REST server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="large-v3-turbo", help="Whisper model name")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, cuda:2, cpu …")
    parser.add_argument(
        "--token2wav-path",
        type=str,
        required=True,
        help="Token2Wav path for CosyVoice model"
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default="Romeo and Juliet might be the most famous act of William Shakespeare.",
        help="The prompt text for CosyVoice"
    )
    parser.add_argument(
        "--prompt-speech-path",
        type=str,
        default="./assets/common_voice_en_2586258.wav",
        help="The path to the prompt speech for CosyVoice"
    )
    
    args, _ = parser.parse_known_args()
    return args

args = get_args()

DEVICE = args.device

# Load Whisper once
print(f"Loading Whisper '{args.model}' on {DEVICE} …")
WHISPER = whisper.load_model(args.model, device=DEVICE).eval()

def _get_mel_bins(model) -> int:
    if hasattr(model, "n_mels"):
        return int(model.n_mels)
    if hasattr(model, "conv1"):
        return int(model.conv1.weight.shape[1])
    if hasattr(model, "encoder") and hasattr(model.encoder, "conv1"):
        return int(model.encoder.conv1.weight.shape[1])
    return 80

REQ_BINS = _get_mel_bins(WHISPER)

# Load CosyVoice2 once
print("Loading CosyVoice2 …")
CODEC = CosyVoice2(
    args.token2wav_path, load_jit=False, load_trt=False, fp16=False
)

# Load prompt speech if provided
PROMPT_SPEECH_16K = None
if args.prompt_speech_path:
    try:
        PROMPT_SPEECH_16K = load_wav(args.prompt_speech_path, 16000)
        print(f"Loaded prompt speech from {args.prompt_speech_path}")
    except Exception as e:
        print(f"Warning: Could not load prompt speech: {e}")
        PROMPT_SPEECH_16K = None

# Tokenizer
TOKENIZER = whisper.tokenizer.get_tokenizer(multilingual=True, task="transcribe")

# ---------------------------------------------------------------------------
# CosyVoice2 audio decoding function
# ---------------------------------------------------------------------------
def audio_decode_cosyvoice2(
    audio_tokens, prompt_text, prompt_speech_16k, codec_decoder
):
    """
    Generate audio from tokens with optional tone and prompt embedding.
    """
    model_inputs_dict = codec_decoder.frontend.frontend_zero_shot(
        "empty", prompt_text, prompt_speech_16k, 24000
    )
    tts_mel, _ = codec_decoder.model.flow.inference(
        token=audio_tokens.to(codec_decoder.model.device),
        token_len=torch.tensor([audio_tokens.shape[1]], dtype=torch.int32).to(
            codec_decoder.model.device
        ),
        prompt_token=model_inputs_dict["flow_prompt_speech_token"].to(
            codec_decoder.model.device
        ),
        prompt_token_len=torch.tensor(
            [model_inputs_dict["flow_prompt_speech_token_len"]], dtype=torch.int32
        ).to(codec_decoder.model.device),
        prompt_feat=model_inputs_dict["prompt_speech_feat"].to(
            codec_decoder.model.device
        ),
        prompt_feat_len=model_inputs_dict["prompt_speech_feat_len"].to(
            codec_decoder.model.device
        ),
        embedding=model_inputs_dict["flow_embedding"].to(codec_decoder.model.device),
        finalize=True,
    )

    audio_hat, _ = codec_decoder.model.hift.inference(
        speech_feat=tts_mel, cache_source=torch.zeros(1, 1, 0)
    )

    return audio_hat

# ---------------------------------------------------------------------------
# FastAPI definitions
# ---------------------------------------------------------------------------
app = FastAPI(title="Whisper NLL server", version="0.1.0")


class ScoreRequest(BaseModel):
    tokens: List[int] = Field(..., description="Speech token ids (<|s_xxx|>)")
    text: str = Field(..., description="Ground‑truth text for NLL")


class ScoreResponse(BaseModel):
    nll: float
    transcript: str


@torch.inference_mode()
def tokens_to_wav(tokens: List[int]) -> torch.Tensor:
    """Convert speech tokens to wav using CosyVoice2"""
    if PROMPT_SPEECH_16K is None:
        raise ValueError("Prompt speech not available. Please provide --prompt-speech-path")
    
    # Convert tokens to tensor format expected by CosyVoice2
    audio_tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)
    
    # Use CosyVoice2 to decode tokens to audio
    audio_hat = audio_decode_cosyvoice2(
        audio_tokens,
        args.prompt_text,
        PROMPT_SPEECH_16K,
        CODEC,
    )
    
    # Return audio as float tensor on CPU
    return audio_hat.squeeze(0).float().cpu()


@torch.inference_mode()
def whisper_nll(wav: torch.Tensor, text: str) -> float:
    mel = _wa.log_mel_spectrogram(_wa.pad_or_trim(wav.numpy()), n_mels=REQ_BINS)
    mel = torch.as_tensor(mel, device=DEVICE)[None]
    tgt = torch.tensor([TOKENIZER.sot] + TOKENIZER.encode(text) + [TOKENIZER.eot], device=DEVICE)[None]
    enc = WHISPER.encoder(mel)
    logits = WHISPER.decoder(tgt[:, :-1], enc)
    return float(cross_entropy(logits.view(-1, logits.size(-1)), tgt[:, 1:].view(-1)))


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest):
    try:
        wav = tokens_to_wav(req.tokens)
        nll_val = whisper_nll(wav, req.text)
        # greedy transcript (quick)
        transcript = WHISPER.transcribe(audio=wav.numpy(), fp16=torch.cuda.is_available())["text"].strip()
        return ScoreResponse(nll=nll_val, transcript=transcript)
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@app.get("/healthz")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
