import torchaudio
import torch


def load_audio(path: str) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if not sr == 24000:
        wav = torchaudio.functional.resample(
            wav,
            sr,
            24000,
        )
    return wav


def get_vocab() -> dict:
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
