from wavtokenizer import get_wavtokenizer_model, get_wavtokenizer_encoded
from device import get_device
from utils import load_audio
import pandas as pd
import json
import os


def save_csv(
    data: list,
    data_folder_path: str,
    filename_id: int,
) -> None:
    pd.DataFrame(data, columns=["text", "audio"]).to_csv(
        f"{data_folder_path}/{filename_id}.csv", index=False
    )


def create_data(
    metadata_path: str,
    data_folder_path: str,
) -> None:
    data = []
    df = pd.read_csv(metadata_path)
    df_len = len(df)
    os.makedirs(data_folder_path, exist_ok=True)
    filename_id = len(os.listdir(data_folder_path))
    df = df.iloc[filename_id * csv_size :]
    for _, row in df.iterrows():
        data.append(
            [
                row.iloc[0],
                get_wavtokenizer_encoded(
                    load_audio(row.iloc[1]).to(device),
                    wavtokenizer,
                )
                .flatten()
                .tolist(),
            ]
        )
        print(
            f"\rProcessing: {filename_id * csv_size + len(data)}/{df_len}",
            end="",
            flush=True,
        )
        if len(data) == csv_size:
            save_csv(
                data,
                data_folder_path,
                filename_id,
            )
            filename_id += 1
            data = []
    if len(data):
        save_csv(
            data,
            data_folder_path,
            filename_id,
        )
    if len(df):
        print("\nDone.")
    else:
        print("Already Done.")


if __name__ == "__main__":
    with open(f"data.json", "r") as f:
        config = json.load(f)

    train_metadata_path = config["train_metadata_path"]
    train_data_folder_path = config["train_data_folder_path"]

    val_metadata_path = config["val_metadata_path"]
    val_data_folder_path = config["val_data_folder_path"]

    csv_size = config["csv_size"]

    device = get_device()

    wavtokenizer = get_wavtokenizer_model()
    wavtokenizer.to(device)
    wavtokenizer.eval()

    print("Creating Training Data...")
    create_data(train_metadata_path, train_data_folder_path)

    print()

    print("Creating Validation Data...")
    create_data(val_metadata_path, val_data_folder_path)
