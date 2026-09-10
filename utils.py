"""Shared helpers: audio loading and the fixed text vocabulary."""

from typing import Dict

import torch
import torchaudio

SAMPLE_RATE = 24000


def load_audio(path: str) -> torch.Tensor:
    """Load a waveform and resample to SAMPLE_RATE if needed."""
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            SAMPLE_RATE,
        )
    return waveform


def get_text_vocab() -> Dict[str, int]:
    """
    Fixed character vocabulary used by TextTokenizer.

    Characters are English-only (a–z, space, and a few punctuation marks),
    plus special tokens. Extend or replace this dict for other languages.
    """
    return {
        "a": 0,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
        "f": 5,
        "g": 6,
        "h": 7,
        "i": 8,
        "j": 9,
        "k": 10,
        "l": 11,
        "m": 12,
        "n": 13,
        "o": 14,
        "p": 15,
        "q": 16,
        "r": 17,
        "s": 18,
        "t": 19,
        "u": 20,
        "v": 21,
        "w": 22,
        "x": 23,
        "y": 24,
        "z": 25,
        " ": 26,
        "?": 27,
        "!": 28,
        ".": 29,
        ",": 30,
        "<pad>": 31,
        "<sos>": 32,
        "<eos>": 33,
        "<unk>": 34,
    }
