"""Thin wrappers around the WavTokenizer encoder/decoder."""

import json

import torch
from decoder.pretrained import WavTokenizer


def load_wavtokenizer() -> WavTokenizer:
    """Load WavTokenizer from paths listed in wavtokenizer.json."""
    with open("wavtokenizer.json", "r") as f:
        config = json.load(f)
    return WavTokenizer.from_pretrained0802(
        config["config_path"],
        config["model_path"],
    )


def encode_waveform(
    waveform: torch.Tensor,
    model: WavTokenizer,
) -> torch.Tensor:
    """Encode a waveform into discrete audio codes."""
    bandwidth_id = torch.tensor(
        [0],
        dtype=torch.long,
        device=waveform.device,
    )
    with torch.no_grad():
        codes = model.encode(
            waveform,
            bandwidth_id=bandwidth_id,
        )[-1]
    return codes


def decode_codes(
    codes: torch.Tensor,
    model: WavTokenizer,
) -> torch.Tensor:
    """Decode discrete audio codes back into a waveform."""
    bandwidth_id = torch.tensor(
        [0],
        dtype=torch.long,
        device=codes.device,
    )
    with torch.no_grad():
        waveform = model.decode(
            model.codes_to_features(codes),
            bandwidth_id=bandwidth_id,
        )
    return waveform
