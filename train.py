from modules import (
    TextTokenizer,
    AudioTokenizer,
    Model,
    Dataset,
)
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from device import get_device
import pandas as pd
import pickle
import torch
import json
import os


if __name__ == "__main__":
    with open("train.json", "r") as f:
        config = json.load(f)

    model_folder_path = config["model_folder_path"]
    os.makedirs(model_folder_path, exist_ok=True)

    if not os.path.isfile(f"{model_folder_path}/model.pth"):
        dim_embed = config["dim_embed"]
        nheads = config["nheads"]
        nlayers = config["nlayers"]
        dim_feedforward = config["dim_feedforward"]
        text_max_len = config["text_max_len"]
        audio_max_len = config["audio_max_len"]
        
        initial_lr = config["initial_lr"]
        nepochs = config["nepochs"]
        batch_size = config["batch_size"]

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
            dim_embed,
            nheads,
            nlayers,
            dim_feedforward,
            text_max_len,
            audio_max_len,
        )

        with open(f"{model_folder_path}/config.json", "w") as f:
            json.dump(
                {
                    "text_vocab_size": text_vocab_size,
                    "audio_vocab_size": audio_vocab_size,
                    "dim_embed": dim_embed,
                    "nheads": nheads,
                    "nlayers": nlayers,
                    "dim_feedforward": dim_feedforward,
                    "text_max_len": text_max_len,
                    "audio_max_len": audio_max_len,
                    "initial_lr": initial_lr,
                    "nepochs": nepochs,
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
            config = json.load(f)

        text_vocab_size = config["text_vocab_size"]
        audio_vocab_size = config["audio_vocab_size"]
        dim_embed = config["dim_embed"]
        nheads = config["nheads"]
        nlayers = config["nlayers"]
        dim_feedforward = config["dim_feedforward"]
        text_max_len = config["text_max_len"]
        audio_max_len = config["audio_max_len"]

        initial_lr = config["initial_lr"]
        nepochs = config["nepochs"]
        batch_size = config["batch_size"]

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
        config = json.load(f)

    train_data_folder_path = config["train_data_folder_path"]
    val_data_folder_path = config["val_data_folder_path"]

    initial_epoch = 0
    if os.path.isfile(f"{model_folder_path}/logs.txt"):
        with open(f"{model_folder_path}/logs.txt", "r") as f:
            initial_epoch = len(f.read().split("\n")) - 1

    for epoch in range(initial_epoch, nepochs):
        print("Training...")
        model.train()
        train_asr_loss = 0
        train_tts_loss = 0
        train_len = 0

        csv_files = os.listdir(
            train_data_folder_path,
        )
        for csv_idx, csv_file in enumerate(csv_files):
            train_df = pd.read_csv(f"{train_data_folder_path}/{csv_file}")
            train_dataset = Dataset(train_df, text_tokenizer, audio_tokenizer)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=os.cpu_count(),
            )

            for text, audio in train_loader:
                text, text_key_padding_mask = [v.to(device) for v in text]
                audio, audio_key_padding_mask = [v.to(device) for v in audio]

                optimizer.zero_grad()

                with autocast(device.type):
                    text_output, _ = model.asr(
                        text[:, :-1],
                        audio,
                        None,
                        torch.concat(
                            [
                                audio_key_padding_mask,
                                text_key_padding_mask[:, :-1],
                            ],
                            dim=1,
                        ),
                    )
                    asr_loss = torch.nn.functional.cross_entropy(
                        text_output.reshape(-1, text_output.shape[-1])[
                            ~text_key_padding_mask[:, 1:].reshape(-1)
                        ],
                        text[:, 1:].reshape(-1)[
                            ~text_key_padding_mask[:, 1:].reshape(-1)
                        ],
                    )
                    audio_output, _ = model.tts(
                        audio[:, :-1],
                        text,
                        None,
                        torch.concat(
                            [
                                text_key_padding_mask,
                                audio_key_padding_mask[:, :-1],
                            ],
                            dim=1,
                        ),
                    )
                    tts_loss = torch.nn.functional.cross_entropy(
                        audio_output.reshape(-1, audio_output.shape[-1])[
                            ~audio_key_padding_mask[:, 1:].reshape(-1)
                        ],
                        audio[:, 1:].reshape(-1)[
                            ~audio_key_padding_mask[:, 1:].reshape(-1)
                        ],
                    )

                loss = asr_loss + tts_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_asr_loss += asr_loss.item()
                train_tts_loss += tts_loss.item()
                train_len += 1

                print(
                    f"\rMean Loss:\t{train_asr_loss/train_len:.4f}\t{train_tts_loss/train_len:.4f}\t[{csv_idx+1}/{len(csv_files)}]\t[{epoch+1}/{nepochs}]",
                    end="",
                    flush=True,
                )

            for gp in optimizer.param_groups:
                gp["lr"] = max(initial_lr / 10, gp["lr"] * 0.995)

            torch.save(model.state_dict(), f"{model_folder_path}/model.pth")
            torch.save(scaler.state_dict(), f"{model_folder_path}/scaler.pth")
            torch.save(optimizer.state_dict(), f"{model_folder_path}/optimizer.pth")

        print("\nValidating...")
        model.eval()
        val_asr_loss = 0
        val_tts_loss = 0
        val_len = 0

        csv_files = os.listdir(
            val_data_folder_path,
        )
        for csv_file in csv_files:
            val_df = pd.read_csv(f"{val_data_folder_path}/{csv_file}")
            val_dataset = Dataset(val_df, text_tokenizer, audio_tokenizer)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=os.cpu_count(),
            )

            for text, audio in val_loader:
                text, text_key_padding_mask = [v.to(device) for v in text]
                audio, audio_key_padding_mask = [v.to(device) for v in audio]

                with torch.no_grad():
                    text_output, _ = model.asr(
                        text[:, :-1],
                        audio,
                        None,
                        torch.concat(
                            [
                                audio_key_padding_mask,
                                text_key_padding_mask[:, :-1],
                            ],
                            dim=1,
                        ),
                    )
                    asr_loss = torch.nn.functional.cross_entropy(
                        text_output.reshape(-1, text_output.shape[-1])[
                            ~text_key_padding_mask[:, 1:].reshape(-1)
                        ],
                        text[:, 1:].reshape(-1)[
                            ~text_key_padding_mask[:, 1:].reshape(-1)
                        ],
                    )
                    audio_output, _ = model.tts(
                        audio[:, :-1],
                        text,
                        None,
                        torch.concat(
                            [
                                text_key_padding_mask,
                                audio_key_padding_mask[:, :-1],
                            ],
                            dim=1,
                        ),
                    )
                    tts_loss = torch.nn.functional.cross_entropy(
                        audio_output.reshape(-1, audio_output.shape[-1])[
                            ~audio_key_padding_mask[:, 1:].reshape(-1)
                        ],
                        audio[:, 1:].reshape(-1)[
                            ~audio_key_padding_mask[:, 1:].reshape(-1)
                        ],
                    )

                val_asr_loss += asr_loss.item()
                val_tts_loss += tts_loss.item()
                val_len += 1

                print(
                    f"\rMean Loss:\t{val_asr_loss/val_len:.4f}\t{val_tts_loss/val_len:.4f}\t[{epoch+1}/{nepochs}]",
                    end="",
                    flush=True,
                )

        with open(f"{model_folder_path}/logs.txt", "a") as f:
            f.write(
                f"{train_asr_loss/train_len:.4f}\t{train_tts_loss/train_len:.4f}\t{val_asr_loss/val_len:.4f}\t{val_tts_loss/val_len:.4f}\t[{epoch+1}/{nepochs}]\n"
            )

        print()
