# STC: Speech and Text Conversion—A Multilingual Unified TTS/ASR Model

This repository contains the code accompanying the paper **[STC: Speech and Text Conversion—A Multilingual Unified TTS/ASR Model](#)**.

## Environment

- Python 3.8.10
- CUDA 12.2
- PyTorch 2.4.1

## Third-party code (WavTokenizer)

The `encoder/` and `decoder/` directories are taken from the [WavTokenizer](https://github.com/jishengpeng/WavTokenizer) repository (MIT License). They provide the discrete audio codec used to encode and decode waveforms in this project. They are included here largely unchanged; please refer to the upstream repo for the original implementation and pretrained checkpoints.

If you use this codebase, please also cite WavTokenizer as appropriate.

## Setup and Usage

### 1. Create metadata CSV files

Before using the training pipeline, create two metadata CSV files (one for training and one for validation). Each file must have:

- **First column:** `text`
- **Second column:** `audio` (path to the corresponding audio file)

Example header row:

```csv
text,audio
```

### 2. Configure data and WavTokenizer paths

Update `data.json` with:

- Correct paths to your train and validation metadata CSVs
- Desired `shard_size` (number of samples per shard CSV)
- Output folder paths for the generated train/val data

Update `wavtokenizer.json` so that the WavTokenizer **model** and **config** file paths are correct.

### 3. Prepare training data

Run:

```bash
python data.py
```

This encodes audio with WavTokenizer and writes the training and validation shard files used by the trainer.

### 4. Train

Edit `train.json` according to your needs (model size, sequence lengths, learning rate, epochs, batch size, output folder, etc.).

**Note:** The text tokenizer uses a **fixed vocabulary** with **English-only** characters (plus a few punctuation marks and special tokens). If you need another language or a different character set, update the vocabulary in `utils.py` (`get_text_vocab`) before training.

Then run:

```bash
python train.py
```

### 5. Inference

After training, configure `inference.json` (checkpoint folder, audio prompt path, optional `text_input`, and output audio path), then run:

```bash
python inference.py
```

## License

This project is released under the MIT License (see [`LICENSE`](LICENSE)). The bundled `encoder/` and `decoder/` code from WavTokenizer is also MIT-licensed; its original copyright notice is retained in the license file.
