"""
Training entrypoint for the joint ASR + TTS model.

Reads hyperparameters from train.json and data paths from data.json.
If model_folder_path already contains model.pth (and related files), training
resumes from the last logged epoch; otherwise a new model and tokenizers are created.
"""

import json
import os
import pickle
from typing import Tuple

import pandas as pd
import torch
from device import get_device
from modules import AudioTokenizer, Model, TextAudioDataset, TextTokenizer
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy over non-padding target positions."""
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = ~target_padding_mask.reshape(-1)
    return torch.nn.functional.cross_entropy(
        flat_logits[flat_mask],
        flat_targets[flat_mask],
    )


def compute_asr_tts_losses(
    model: Model,
    text: torch.Tensor,
    text_padding_mask: torch.Tensor,
    audio: torch.Tensor,
    audio_padding_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced ASR and TTS losses for one batch."""
    text_logits, _ = model.asr(
        text[:, :-1],
        audio,
        None,
        torch.concat([audio_padding_mask, text_padding_mask[:, :-1]], dim=1),
    )
    asr_loss = masked_cross_entropy(
        text_logits,
        text[:, 1:],
        text_padding_mask[:, 1:],
    )

    audio_logits, _ = model.tts(
        audio[:, :-1],
        text,
        None,
        torch.concat([text_padding_mask, audio_padding_mask[:, :-1]], dim=1),
    )
    tts_loss = masked_cross_entropy(
        audio_logits,
        audio[:, 1:],
        audio_padding_mask[:, 1:],
    )
    return asr_loss, tts_loss


def save_training_state(
    model_folder_path: str,
    model: Model,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> None:
    """Persist model, optimizer, and amp scaler checkpoints."""
    torch.save(model.state_dict(), f"{model_folder_path}/model.pth")
    torch.save(optimizer.state_dict(), f"{model_folder_path}/optimizer.pth")
    torch.save(scaler.state_dict(), f"{model_folder_path}/scaler.pth")


if __name__ == "__main__":
    with open("train.json", "r") as f:
        train_config = json.load(f)

    model_folder_path = train_config["model_folder_path"]
    os.makedirs(model_folder_path, exist_ok=True)

    # --- Initialize or resume model / tokenizers ---
    if not os.path.isfile(f"{model_folder_path}/model.pth"):
        embed_dim = train_config["embed_dim"]
        num_heads = train_config["num_heads"]
        num_layers = train_config["num_layers"]
        feedforward_dim = train_config["feedforward_dim"]
        text_max_len = train_config["text_max_len"]
        audio_max_len = train_config["audio_max_len"]
        initial_lr = train_config["initial_lr"]
        num_epochs = train_config["num_epochs"]
        batch_size = train_config["batch_size"]

        text_tokenizer = TextTokenizer(text_max_len)
        text_vocab_size = text_tokenizer.vocab_size
        with open(f"{model_folder_path}/text_tokenizer.pkl", "wb") as f:
            pickle.dump(text_tokenizer, f)

        audio_tokenizer = AudioTokenizer(audio_max_len)
        audio_vocab_size = audio_tokenizer.vocab_size
        with open(f"{model_folder_path}/audio_tokenizer.pkl", "wb") as f:
            pickle.dump(audio_tokenizer, f)

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

        with open(f"{model_folder_path}/config.json", "w") as f:
            json.dump(
                {
                    "text_vocab_size": text_vocab_size,
                    "audio_vocab_size": audio_vocab_size,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "feedforward_dim": feedforward_dim,
                    "text_max_len": text_max_len,
                    "audio_max_len": audio_max_len,
                    "initial_lr": initial_lr,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                },
                f,
            )
    else:
        with open(f"{model_folder_path}/text_tokenizer.pkl", "rb") as f:
            text_tokenizer = pickle.load(f)

        with open(f"{model_folder_path}/audio_tokenizer.pkl", "rb") as f:
            audio_tokenizer = pickle.load(f)

        with open(f"{model_folder_path}/config.json", "r") as f:
            saved_config = json.load(f)

        text_vocab_size = saved_config["text_vocab_size"]
        audio_vocab_size = saved_config["audio_vocab_size"]
        embed_dim = saved_config["embed_dim"]
        num_heads = saved_config["num_heads"]
        num_layers = saved_config["num_layers"]
        feedforward_dim = saved_config["feedforward_dim"]
        text_max_len = saved_config["text_max_len"]
        audio_max_len = saved_config["audio_max_len"]
        initial_lr = saved_config["initial_lr"]
        num_epochs = saved_config["num_epochs"]
        batch_size = saved_config["batch_size"]

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
        model.load_state_dict(torch.load(f"{model_folder_path}/model.pth"))

    device = get_device()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    if os.path.isfile(f"{model_folder_path}/optimizer.pth"):
        optimizer.load_state_dict(torch.load(f"{model_folder_path}/optimizer.pth"))

    scaler = GradScaler(device.type)
    if os.path.isfile(f"{model_folder_path}/scaler.pth"):
        scaler.load_state_dict(torch.load(f"{model_folder_path}/scaler.pth"))

    with open("data.json", "r") as f:
        data_config = json.load(f)

    train_data_folder_path = data_config["train_data_folder_path"]
    val_data_folder_path = data_config["val_data_folder_path"]

    # Resume from the number of completed epochs recorded in logs.txt.
    start_epoch = 0
    if os.path.isfile(f"{model_folder_path}/logs.txt"):
        with open(f"{model_folder_path}/logs.txt", "r") as f:
            start_epoch = len(f.read().split("\n")) - 1

    for epoch in range(start_epoch, num_epochs):
        # --- Training ---
        print("Training...")
        model.train()
        train_asr_loss_sum = 0.0
        train_tts_loss_sum = 0.0
        train_steps = 0

        train_shards = os.listdir(train_data_folder_path)
        for shard_index, shard_file in enumerate(train_shards):
            train_df = pd.read_csv(f"{train_data_folder_path}/{shard_file}")
            train_loader = DataLoader(
                TextAudioDataset(train_df, text_tokenizer, audio_tokenizer),
                batch_size=batch_size,
                shuffle=True,
                num_workers=os.cpu_count(),
            )

            for text_batch, audio_batch in train_loader:
                text, text_padding_mask = [tensor.to(device) for tensor in text_batch]
                audio, audio_padding_mask = [
                    tensor.to(device) for tensor in audio_batch
                ]

                optimizer.zero_grad()
                with autocast(device.type):
                    asr_loss, tts_loss = compute_asr_tts_losses(
                        model,
                        text,
                        text_padding_mask,
                        audio,
                        audio_padding_mask,
                    )
                    loss = asr_loss + tts_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_asr_loss_sum += asr_loss.item()
                train_tts_loss_sum += tts_loss.item()
                train_steps += 1

                print(
                    f"\rMean Loss:\t{train_asr_loss_sum / train_steps:.4f}\t"
                    f"{train_tts_loss_sum / train_steps:.4f}\t"
                    f"[{shard_index + 1}/{len(train_shards)}]\t"
                    f"[{epoch + 1}/{num_epochs}]",
                    end="",
                    flush=True,
                )

            # Decay LR after each train shard; floor at initial_lr / 10.
            for param_group in optimizer.param_groups:
                param_group["lr"] = max(initial_lr / 10, param_group["lr"] * 0.995)

            save_training_state(model_folder_path, model, optimizer, scaler)

        # --- Validation ---
        print("\nValidating...")
        model.eval()
        val_asr_loss_sum = 0.0
        val_tts_loss_sum = 0.0
        val_steps = 0

        val_shards = os.listdir(val_data_folder_path)
        for shard_file in val_shards:
            val_df = pd.read_csv(f"{val_data_folder_path}/{shard_file}")
            val_loader = DataLoader(
                TextAudioDataset(val_df, text_tokenizer, audio_tokenizer),
                batch_size=batch_size,
                num_workers=os.cpu_count(),
            )

            for text_batch, audio_batch in val_loader:
                text, text_padding_mask = [tensor.to(device) for tensor in text_batch]
                audio, audio_padding_mask = [
                    tensor.to(device) for tensor in audio_batch
                ]

                with torch.no_grad():
                    asr_loss, tts_loss = compute_asr_tts_losses(
                        model,
                        text,
                        text_padding_mask,
                        audio,
                        audio_padding_mask,
                    )

                val_asr_loss_sum += asr_loss.item()
                val_tts_loss_sum += tts_loss.item()
                val_steps += 1

                print(
                    f"\rMean Loss:\t{val_asr_loss_sum / val_steps:.4f}\t"
                    f"{val_tts_loss_sum / val_steps:.4f}\t"
                    f"[{epoch + 1}/{num_epochs}]",
                    end="",
                    flush=True,
                )

        with open(f"{model_folder_path}/logs.txt", "a") as f:
            f.write(
                f"{train_asr_loss_sum / train_steps:.4f}\t"
                f"{train_tts_loss_sum / train_steps:.4f}\t"
                f"{val_asr_loss_sum / val_steps:.4f}\t"
                f"{val_tts_loss_sum / val_steps:.4f}\t"
                f"[{epoch + 1}/{num_epochs}]\n"
            )

        print()
