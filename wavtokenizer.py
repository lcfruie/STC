from decoder.pretrained import WavTokenizer
import torch
import json


def get_wavtokenizer_model() -> WavTokenizer:
    with open(f"wavtokenizer.json", "r") as f:
        config = json.load(f)
    return WavTokenizer.from_pretrained0802(
        config["config_path"],
        config["model_path"],
    )


def get_wavtokenizer_encoded(
    x: torch.Tensor,
    model: WavTokenizer,
) -> torch.Tensor:
    bandwidth_id = torch.tensor(
        [0],
        dtype=torch.long,
        device=x.device,
    )
    with torch.no_grad():
        x = model.encode(
            x,
            bandwidth_id=bandwidth_id,
        )[-1]
    return x


def get_wavtokenizer_decoded(
    x: torch.Tensor,
    model: WavTokenizer,
) -> torch.Tensor:
    bandwidth_id = torch.tensor(
        [0],
        dtype=torch.long,
        device=x.device,
    )
    with torch.no_grad():
        x = model.decode(
            model.codes_to_features(x),
            bandwidth_id=bandwidth_id,
        )
    return x
