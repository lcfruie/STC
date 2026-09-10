"""
Build training/validation shards from metadata CSVs.

Reads paths and shard size from data.json, encodes each audio file with
WavTokenizer, and writes chunked CSV files under the configured data folders.
Each output CSV has columns: text, audio (list of discrete token ids).
"""

import json
import os

import pandas as pd
import torch
from decoder.pretrained import WavTokenizer
from device import get_device
from utils import load_audio
from wavtokenizer import encode_waveform, load_wavtokenizer


def save_shard(
    rows: list,
    output_dir: str,
    shard_id: int,
) -> None:
    """Write one shard of (text, audio-token) pairs as a CSV."""
    pd.DataFrame(rows, columns=["text", "audio"]).to_csv(
        f"{output_dir}/{shard_id}.csv",
        index=False,
    )


def create_sharded_dataset(
    metadata_path: str,
    output_dir: str,
    shard_size: int,
    device: torch.device,
    wavtokenizer: WavTokenizer,
) -> None:
    """
    Encode metadata rows into sharded CSVs under output_dir.

    Resumes from the number of existing shard files so interrupted runs can
    continue.
    """
    metadata = pd.read_csv(metadata_path)
    total_rows = len(metadata)
    os.makedirs(output_dir, exist_ok=True)

    # Skip rows already covered by existing shard files.
    shard_id = len(os.listdir(output_dir))
    metadata = metadata.iloc[shard_id * shard_size :]

    shard_rows = []
    for _, row in metadata.iterrows():
        text = row.iloc[0]
        audio_path = row.iloc[1]
        audio_codes = (
            encode_waveform(
                load_audio(audio_path).to(device),
                wavtokenizer,
            )
            .flatten()
            .tolist()
        )
        shard_rows.append([text, audio_codes])

        processed = shard_id * shard_size + len(shard_rows)
        print(f"\rProcessing: {processed}/{total_rows}", end="", flush=True)

        if len(shard_rows) == shard_size:
            save_shard(shard_rows, output_dir, shard_id)
            shard_id += 1
            shard_rows = []

    if shard_rows:
        save_shard(shard_rows, output_dir, shard_id)

    if len(metadata) > 0:
        print("\nDone.")
    else:
        print("Already Done.")


if __name__ == "__main__":
    with open("data.json", "r") as f:
        config = json.load(f)

    train_metadata_path = config["train_metadata_path"]
    train_data_folder_path = config["train_data_folder_path"]
    val_metadata_path = config["val_metadata_path"]
    val_data_folder_path = config["val_data_folder_path"]
    shard_size = config["shard_size"]

    device = get_device()
    wavtokenizer = load_wavtokenizer()
    wavtokenizer.to(device)
    wavtokenizer.eval()

    print("Creating Training Data...")
    create_sharded_dataset(
        train_metadata_path,
        train_data_folder_path,
        shard_size,
        device,
        wavtokenizer,
    )

    print()

    print("Creating Validation Data...")
    create_sharded_dataset(
        val_metadata_path,
        val_data_folder_path,
        shard_size,
        device,
        wavtokenizer,
    )
