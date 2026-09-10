"""
Inference entrypoint for ASR (audio prompt -> text) and optional TTS.

Loads a trained checkpoint from model_folder_path (see inference.json),
transcribes the audio prompt, then — if text_input and audio_output_path
are set — continues generation with TTS and writes a waveform.
"""

import json
import pickle

import torch
import torchaudio
from device import get_device
from modules import AUDIO_CODEBOOK_SIZE, AudioTokenizer, Model, TextTokenizer
from utils import SAMPLE_RATE, load_audio
from wavtokenizer import decode_codes, encode_waveform, load_wavtokenizer


def filter_logits(
    logits: torch.Tensor,
    top_k: int = 200,
    top_p: float = 0.90,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply temperature, top-k, and nucleus (top-p) filtering to logits."""
    filtered = logits.clone() / temperature
    filter_value = -float("Inf")

    if top_k > 0:
        top_k_threshold = torch.topk(filtered, top_k, dim=-1)[0][..., -1, None]
        filtered[filtered < top_k_threshold] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(
            sorted_logits - sorted_logits.max(dim=-1, keepdim=True)[0],
            dim=-1,
        )
        cumulative_probs = sorted_probs.cumsum(dim=-1)

        remove_sorted = cumulative_probs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False

        remove_mask = torch.zeros_like(filtered, dtype=torch.bool)
        remove_mask.scatter_(1, sorted_indices, remove_sorted)
        filtered[remove_mask] = filter_value

    return filtered


if __name__ == "__main__":
    with open("inference.json", "r") as f:
        inference_config = json.load(f)

    model_folder_path = inference_config["model_folder_path"]
    audio_prompt_path = inference_config["audio_prompt_path"]
    audio_output_path = inference_config["audio_output_path"]
    text_input = inference_config["text_input"]

    # --- Load tokenizers, config, and checkpoint ---
    with open(f"{model_folder_path}/text_tokenizer.pkl", "rb") as f:
        text_tokenizer: TextTokenizer = pickle.load(f)

    with open(f"{model_folder_path}/audio_tokenizer.pkl", "rb") as f:
        audio_tokenizer: AudioTokenizer = pickle.load(f)

    with open(f"{model_folder_path}/config.json", "r") as f:
        model_config = json.load(f)

    text_vocab_size = model_config["text_vocab_size"]
    audio_vocab_size = model_config["audio_vocab_size"]
    embed_dim = model_config["embed_dim"]
    num_heads = model_config["num_heads"]
    num_layers = model_config["num_layers"]
    feedforward_dim = model_config["feedforward_dim"]
    text_max_len = model_config["text_max_len"]
    audio_max_len = model_config["audio_max_len"]

    model = Model(
        text_vocab_size,
        audio_vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        text_max_len,
        audio_max_len,
    )
    model.load_state_dict(
        torch.load(f"{model_folder_path}/model.pth", map_location=torch.device("cpu"))
    )
    device = get_device()
    model.to(device)
    model.eval()

    wavtokenizer = load_wavtokenizer()
    wavtokenizer.to(device)
    wavtokenizer.eval()

    prompt_codes = encode_waveform(
        load_audio(audio_prompt_path).to(device),
        wavtokenizer,
    )
    prompt_codes = prompt_codes.detach().cpu().flatten()

    text_tokens = text_tokenizer.encode().unsqueeze(0).to(device)
    audio_tokens = audio_tokenizer.encode(prompt_codes).unsqueeze(0).to(device)

    # --- ASR: autoregressive text generation conditioned on the prompt ---
    with torch.no_grad():
        cache = None
        next_text_token = text_tokens
        while text_tokens.size(1) < text_max_len:
            if cache is None:
                next_text_logits, cache = model.asr(
                    next_text_token,
                    audio_tokens,
                    cache,
                )
            else:
                next_text_logits, cache = model.asr(next_text_token, None, cache)

            next_text_token = torch.argmax(
                next_text_logits[:, -1],
                dim=-1,
            ).unsqueeze(1)

            if next_text_token.item() == text_tokenizer.eos_tok:
                break
            text_tokens = torch.concat([text_tokens, next_text_token], dim=-1)

    transcribed_text = text_tokenizer.decode(text_tokens.squeeze(0))
    print(f"Text: {transcribed_text}")

    # --- Optional TTS: continue from ASR text + text_input ---
    if audio_output_path and text_input:
        tts_text = transcribed_text + "," + text_input
        text_tokens = text_tokenizer.encode(tts_text).unsqueeze(0).to(device)

        # Drop eos from the prompt so generation can continue.
        audio_tokens = audio_tokens[:, :-1]
        prompt_token_len = audio_tokens.size(1)

        with torch.no_grad():
            cache = None
            next_audio_token = audio_tokens
            while audio_tokens.size(1) < audio_max_len:
                if cache is None:
                    next_audio_logits, cache = model.tts(
                        next_audio_token,
                        text_tokens,
                        cache,
                    )
                else:
                    next_audio_logits, cache = model.tts(
                        next_audio_token,
                        None,
                        cache,
                    )

                next_audio_token = torch.multinomial(
                    torch.softmax(
                        filter_logits(next_audio_logits[:, -1]),
                        dim=-1,
                    ),
                    num_samples=1,
                )

                if next_audio_token.item() == audio_tokenizer.eos_tok:
                    break
                audio_tokens = torch.concat([audio_tokens, next_audio_token], dim=-1)

        generated_codes = audio_tokens[:, prompt_token_len:]
        # Keep only codebook ids; zero out special tokens.
        generated_codes = generated_codes * (generated_codes < AUDIO_CODEBOOK_SIZE)

        waveform = decode_codes(generated_codes, wavtokenizer)
        torchaudio.save(
            audio_output_path,
            waveform.detach().cpu(),
            SAMPLE_RATE,
        )
        print("Audio Generated.")
