from modules import (
    TextTokenizer,
    AudioTokenizer,
    Model,
)
from wavtokenizer import (
    get_wavtokenizer_model,
    get_wavtokenizer_encoded,
    get_wavtokenizer_decoded,
)
from device import get_device
from utils import load_audio
import torchaudio
import pickle
import torch
import json


def get_filtered(
    logits: torch.Tensor,
    top_k: int = 200,
    top_p: float = 0.90,
    temperature: float = 1.0,
) -> torch.Tensor:
    logits = logits.clone() / temperature
    if top_k > 0:
        kth_values = torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
        logits[logits < kth_values] = -float("Inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(
            sorted_logits - sorted_logits.max(dim=-1, keepdim=True)[0], dim=-1
        )
        sorted_indices_to_remove = sorted_probs.cumsum(dim=-1) > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float("Inf")
    return logits


if __name__ == "__main__":
    with open("inference.json", "r") as f:
        config = json.load(f)

    model_folder_path = config["model_folder_path"]
    audio_prompt_path = config["audio_prompt_path"]
    audio_output_path = config["audio_output_path"]
    text_input = config["text_input"]

    with open(f"{model_folder_path}/text_tokenizer.pkl", "rb") as f:
        text_tokenizer: TextTokenizer = pickle.load(f)

    with open(f"{model_folder_path}/audio_tokenizer.pkl", "rb") as f:
        audio_tokenizer: AudioTokenizer = pickle.load(f)

    with open(f"{model_folder_path}/config.json", "r") as f:
        config = json.load(f)

    text_vocab_size = config["text_vocab_size"]
    audio_vocab_size = config["audio_vocab_size"]
    dim_embed = config["dim_embed"]
    nheads = config["nheads"]
    nlayers = config["nlayers"]
    dim_feedforward = config["dim_feedforward"]
    text_max_len = config["text_max_len"]
    audio_max_len = config["audio_max_len"]

    model = Model(
        text_vocab_size,
        audio_vocab_size,
        dim_embed,
        nheads,
        nlayers,
        dim_feedforward,
        text_max_len,
        audio_max_len,
    )
    model.load_state_dict(
        torch.load(f"{model_folder_path}/model.pth", map_location=torch.device("cpu"))
    )
    device = get_device()
    model.to(device)
    model.eval()

    wavtokenizer = get_wavtokenizer_model()
    wavtokenizer.to(device)
    wavtokenizer.eval()

    audio = get_wavtokenizer_encoded(
        load_audio(audio_prompt_path).to(device),
        wavtokenizer,
    )
    audio = audio.detach().cpu().flatten()

    text = text_tokenizer.encode()
    text = text.unsqueeze(0).to(device)
    audio = audio_tokenizer.encode(audio)
    audio = audio.unsqueeze(0).to(device)

    with torch.no_grad():
        cache = None
        while text.size(1) < text_max_len:
            if cache == None:
                text_last, cache = model.asr(
                    text,
                    audio,
                    cache,
                )
            else:
                text_last, cache = model.asr(
                    text_last,
                    None,
                    cache,
                )
            text_last = torch.argmax(text_last[:, -1], dim=-1).unsqueeze(1)
            if text_last.item() == text_tokenizer.eos_tok:
                break
            text = torch.concat([text, text_last], dim=-1)

    text = text.squeeze(0)
    text = text_tokenizer.decode(text)

    print(f"Text: {text}")

    if audio_output_path and text_input:
        text = text + "," + text_input
        text = text_tokenizer.encode(text)
        text = text.unsqueeze(0).to(device)

        audio = audio[:, :-1]
        audio_len = audio.size(1)

        with torch.no_grad():
            cache = None
            while audio.size(1) < audio_max_len:
                if cache == None:
                    audio_last, cache = model.tts(
                        audio,
                        text,
                        cache,
                    )
                else:
                    audio_last, cache = model.tts(
                        audio_last,
                        None,
                        cache,
                    )
                audio_last = torch.multinomial(
                    torch.softmax(
                        get_filtered(audio_last[:, -1]),
                        dim=-1,
                    ),
                    num_samples=1,
                )
                if audio_last.item() == audio_tokenizer.eos_tok:
                    break
                audio = torch.concat([audio, audio_last], dim=-1)

        audio = audio[:, audio_len:]
        audio = audio * (audio < 4096)

        audio = get_wavtokenizer_decoded(
            audio,
            wavtokenizer,
        )

        audio = audio.detach().cpu()
        torchaudio.save(audio_output_path, audio, 24000)
        print("Audio Generated.")
