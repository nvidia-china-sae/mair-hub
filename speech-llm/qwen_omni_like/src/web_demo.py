# Modified from https://github.com/QwenLM/Qwen2.5-Omni/blob/main/web_demo.py
import io
import sys
from argparse import ArgumentParser

import gradio as gr
import gradio.processing_utils as processing_utils
import numpy as np
import sherpa_onnx
import soundfile as sf
import torch
import whisper
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from gradio_client import utils as client_utils
from model import SPEECH_LLM, EncoderProjector
from peft import LoraConfig, get_peft_model
from train import DEFAULT_SPEECH_TOKEN, add_model_arguments, add_training_arguments, get_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from whisper_encoder_forward_monkey_patch import replace_whisper_encoder_forward

# https://github.com/FunAudioLLM/CosyVoice/tree/main/third_party
sys.path.append("/workspace/CosyVoice/third_party/Matcha-TTS")

def audio_decode_cosyvoice2(
    audio_tokens, prompt_text, prompt_speech_16k, codec_decoder
):
    """
    Generate audio from tokens with optional tone and prompt embedding.

    Args:
        audio_tokens (list): List of audio tokens to be processed.
        model_config: Configuration object containing vocab settings.
        codec_decoder: Codec decoder for generating audio.
        tone_dir (str): The tone directory or setting.
        audio_prompt_path (str, optional): Path to the audio prompt file. Required when tone_dir is not "default_tone".
        code_layer (int, optional): Number of code layers. Defaults to 1.
        num_latency_tokens (int, optional): Number of latency tokens to ignore. Defaults to 0.
        speed (float, optional): Speed factor for audio generation. Defaults to 1.0.

    Returns:
        torch.Tensor: Generated audio waveform.
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


def _launch_demo(args, model, tokenizer, token2wav_model, asr_model):
    def format_history(history: list):
        messages = []
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item["role"], "content": item["content"]})
        return messages

    def decode(
        model,
        token2wav_model,
        tokenizer,
        feature,
        messages,
    ):
        """Decode one
        Returns:
            pass
        """

        dtype = torch.float32
        device = model.llm.device

        feature = feature.to(device, dtype=dtype)

        input_ids, attention_mask = preprocess([messages], tokenizer)

        generated_ids, audio_tokens = model.decode_with_speech_output(
            feature, input_ids.to(device, dtype=torch.long), attention_mask.to(device)
        )

        hyps = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        yield {"type": "text", "data": hyps[0]}

        audio_tokens = torch.tensor(audio_tokens, dtype=torch.int32).unsqueeze(0)

        # audio_hat = audio_decode_cosyvoice(audio_tokens, token2wav_model)
        prompt_speech_16k = load_wav(args.prompt_speech_path, 16000)
        audio_hat = audio_decode_cosyvoice2(
            audio_tokens,
            args.prompt_text,
            prompt_speech_16k,
            token2wav_model,
        )
        
        audio = audio_hat.squeeze(0).cpu().numpy()
        audio = np.array(audio * 32767).astype(np.int16)
        wav_io = io.BytesIO()
        sf.write(wav_io, audio, samplerate=24000, format="WAV") # 22050 for CosyVoice1
        wav_io.seek(0)
        wav_bytes = wav_io.getvalue()
        audio_path = processing_utils.save_bytes_to_cache(
            wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE
        )

        yield {"type": "audio", "data": audio_path}

    def media_predict(audio, history):
        # First yield
        yield (
            None,  # microphone
            history,  # media_chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        history.append({"role": "user", "content": (audio,)})
        history.append({"role": "user", "content": f"{DEFAULT_SPEECH_TOKEN}"})
        history.append({"role": "assistant", "content": ""})
        formatted_history = format_history(
            history=history
        )  # only keep string text format
        # FIXME: TODO: using multi-rounds en dataset
        formatted_history = formatted_history[-2:]

        assert audio is not None
        audio_transcript = get_transcript(
            audio,
            asr_model,
        )
        history[-2]["content"] = audio_transcript

        fbank = whisper.log_mel_spectrogram(audio, device=model.llm.device)
        fbank = fbank.unsqueeze(0)
        assert fbank.ndim == 3

        for chunk in decode(
            model, token2wav_model, tokenizer, fbank, formatted_history
        ):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # microphone
                    history,  # media_chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append(
                    {"role": "assistant", "content": gr.Audio(chunk["data"])}
                )

        # Final yield
        yield (
            None,  # microphone
            history,  # media_chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    with gr.Blocks() as demo:
        with gr.Tab("Online"):
            with gr.Row():
                with gr.Column(scale=1):
                    microphone = gr.Audio(sources=["microphone"], type="filepath")
                    submit_btn = gr.Button("Submit", variant="primary")
                    stop_btn = gr.Button("Stop", visible=False)
                    clear_btn = gr.Button("Clear History")
                with gr.Column(scale=2):
                    media_chatbot = gr.Chatbot(height=650, type="messages")

                def clear_history():
                    return [], gr.update(value=None)

                submit_event = submit_btn.click(
                    fn=media_predict,
                    inputs=[
                        microphone,
                        media_chatbot,
                    ],
                    outputs=[microphone, media_chatbot, submit_btn, stop_btn],
                )
                stop_btn.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=[submit_event],
                    queue=False,
                )
                clear_btn.click(
                    fn=clear_history, inputs=None, outputs=[media_chatbot, microphone]
                )

    demo.queue(default_concurrency_limit=100, max_size=100).launch(
        max_threads=100,
        ssr_mode=False,
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def _get_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--token2wav-path",
        type=str,
        default=None,
        help="Token2Wav path, default to %(default)r",
    )
    parser.add_argument(
        "--asr-model-dir",
        type=str,
        default=None,
        help="ASR model dir, default to %(default)r",
    )
    parser.add_argument(
        "--flash-attn2",
        action="store_true",
        default=False,
        help="Enable flash_attention_2 when loading the model.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Create a publicly shareable link for the interface.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=False,
        help="Automatically launch the interface in a new tab on the default browser.",
    )
    parser.add_argument(
        "--server-port", type=int, default=8001, help="Demo server port."
    )
    parser.add_argument(
        "--server-name", type=str, default="127.0.0.1", help="Demo server name."
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


def read_wave(wave_filename: str):
    """
    Args:
      wave_filename:
        Path to a wave file. It should be single channel and can be of type
        32-bit floating point PCM. Its sample rate does not need to be 24kHz.

    Returns:
      Return a tuple containing:
       - A 1-D array of dtype np.float32 containing the samples,
         which are normalized to the range [-1, 1].
       - Sample rate of the wave file.
    """

    samples, sample_rate = sf.read(wave_filename, dtype="float32")
    assert (
        samples.ndim == 1
    ), f"Expected single channel, but got {samples.ndim} channels."

    samples_float32 = samples.astype(np.float32)

    return samples_float32, sample_rate


def get_transcript(audio_path, recognizer):
    samples, sample_rate = read_wave(audio_path)
    s = recognizer.create_stream()
    s.accept_waveform(sample_rate, samples)
    recognizer.decode_streams([s])
    return s.result.text


if __name__ == "__main__":
    args = _get_args()
    model, tokenizer = get_model(args)
    model.eval()
    model.to(torch.device("cuda"))
    # token2wav = CosyVoice(
    #     args.token2wav_path, load_jit=False, load_trt=False, fp16=False
    # )
    token2wav = CosyVoice2(
        args.token2wav_path, load_jit=False, load_trt=False, fp16=False
    )

    # TODO: using an English ASR model, also, train model with InstructS2S-200K multi-rounds dataset
    asr_model = sherpa_onnx.OfflineRecognizer.from_paraformer(
        paraformer=f"{args.asr_model_dir}/model.int8.onnx",
        tokens=f"{args.asr_model_dir}/tokens.txt",
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        debug=False,
    )

    _launch_demo(args, model, tokenizer, token2wav, asr_model)
