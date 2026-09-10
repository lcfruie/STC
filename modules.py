"""
Core model pieces: text/audio tokenizers, transformer stack, and dataset.

TextTokenizer uses a fixed character vocabulary from utils.get_text_vocab().
AudioTokenizer wraps WavTokenizer codebook ids (0–4095) plus pad/sos/eos.
Model supports joint ASR and TTS via a shared decoder-only transformer.

Note: tokenizer attribute names (pad_tok, sos_tok, …) are kept stable so
existing pickled tokenizers remain loadable.
"""

import ast
import math
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
from utils import get_text_vocab

# WavTokenizer codebook size; ids >= this value are special tokens.
AUDIO_CODEBOOK_SIZE = 4096


class TextTokenizer:
    """Character-level text tokenizer with sos/eos/pad/unk special tokens."""

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.vocab = get_text_vocab()
        self.pad_tok = self.vocab["<pad>"]
        self.sos_tok = self.vocab["<sos>"]
        self.eos_tok = self.vocab["<eos>"]
        self.unk_tok = self.vocab["<unk>"]
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {token_id: char for char, token_id in self.vocab.items()}

    def encode(
        self,
        text: Optional[str] = None,
        training: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a string to token ids.

        If text is None, returns only [<sos>] (used to start ASR decoding).
        When training=True, pads to max_len and also returns a padding mask.
        """
        if text is None:
            return torch.tensor([self.sos_tok])

        token_ids = [self.vocab.get(char, self.unk_tok) for char in text.lower()]
        tokens = torch.tensor(
            [self.sos_tok] + token_ids[: self.max_len - 2] + [self.eos_tok],
        )

        if training:
            padding = torch.full(
                (self.max_len - tokens.size(0),),
                self.pad_tok,
            )
            tokens = torch.concat((tokens, padding))
            padding_mask = tokens == self.pad_tok
            return tokens, padding_mask

        return tokens

    def decode(self, tokens: torch.Tensor) -> str:
        """Map token ids back to a string, dropping special tokens."""
        special_tokens = {
            self.pad_tok,
            self.sos_tok,
            self.eos_tok,
            self.unk_tok,
        }
        return "".join(
            self.inv_vocab[token_id]
            for token_id in tokens.tolist()
            if token_id not in special_tokens
        )


class AudioTokenizer:
    """
    Discrete audio-code tokenizer.

    Codebook ids are 0–4095; pad=4096, sos=4097, eos=4098.
    """

    def __init__(self, max_len: int):
        self.max_len = max_len
        self.pad_tok = AUDIO_CODEBOOK_SIZE
        self.sos_tok = AUDIO_CODEBOOK_SIZE + 1
        self.eos_tok = AUDIO_CODEBOOK_SIZE + 2
        self.vocab_size = AUDIO_CODEBOOK_SIZE + 3

    def encode(
        self,
        codes: Optional[Union[torch.Tensor, str]] = None,
        training: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode audio codes (tensor or stringified list) with sos/eos.

        If codes is None, returns only [<sos>]. When training=True, pads to
        max_len and returns a padding mask as well.
        """
        if codes is None:
            return torch.tensor([self.sos_tok])

        if isinstance(codes, str):
            codes = torch.tensor(ast.literal_eval(codes))

        tokens = torch.concat(
            [
                torch.tensor([self.sos_tok]),
                codes[: self.max_len - 2],
                torch.tensor([self.eos_tok]),
            ]
        )

        if training:
            padding = torch.full(
                (self.max_len - tokens.size(0),),
                self.pad_tok,
            )
            tokens = torch.concat([tokens, padding])
            padding_mask = tokens == self.pad_tok
            return tokens, padding_mask

        return tokens


def build_positional_embeddings(
    max_len: int,
    embed_dim: int,
) -> torch.Tensor:
    """Sinusoidal positional embeddings of shape (1, max_len, embed_dim)."""
    positions = torch.arange(0, max_len).unsqueeze(1)
    angles = torch.arange(0, embed_dim, 2)
    angles = torch.exp(angles * -(math.log(10000) / embed_dim))

    embeddings = torch.zeros(max_len, embed_dim)
    embeddings[:, 0::2] = torch.sin(positions * angles)
    embeddings[:, 1::2] = torch.cos(positions * angles)
    return embeddings.unsqueeze(0)


class TransformerLayer(nn.Module):
    """Pre-norm transformer block with optional KV cache for autoregressive use."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=batch_first,
        )
        self.activation = activation
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.feedforward_dropout1 = nn.Dropout(dropout)
        self.feedforward_dropout2 = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        self.feedforward_linear1 = nn.Linear(embed_dim, feedforward_dim)
        self.feedforward_linear2 = nn.Linear(feedforward_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.self_attn_norm(hidden_states)
        key_value = (
            query
            if cache is None
            else torch.concat([cache, query], dim=1)
        )
        hidden_states = hidden_states + self._self_attention(
            query=query,
            key_value=key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        hidden_states = hidden_states + self._feedforward(
            self.feedforward_norm(hidden_states)
        )
        return hidden_states, key_value

    def _self_attention(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_output, _ = self.self_attn(
            query=query,
            key=key_value,
            value=key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )
        return self.self_attn_dropout(attn_output)

    def _feedforward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.feedforward_linear1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.feedforward_dropout1(hidden_states)
        hidden_states = self.feedforward_linear2(hidden_states)
        return self.feedforward_dropout2(hidden_states)


class Model(nn.Module):
    """
    Shared transformer for joint ASR and TTS.

    ASR: audio context + text prefix -> next text tokens.
    TTS: text context + audio prefix -> next audio codes.
    """

    def __init__(
        self,
        text_vocab_size: int,
        audio_vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        text_max_len: int,
        audio_max_len: int,
    ):
        super().__init__()
        self.text_embed = nn.Embedding(text_vocab_size, embed_dim)
        self.audio_embed = nn.Embedding(audio_vocab_size, embed_dim)
        self.transformer = nn.ModuleList(
            [
                TransformerLayer(embed_dim, num_heads, feedforward_dim)
                for _ in range(num_layers)
            ]
        )
        # Attribute names match checkpoint keys (norm_fc / text_fc / audio_fc).
        self.norm_fc = nn.LayerNorm(embed_dim)
        self.text_fc = nn.Linear(embed_dim, text_vocab_size)
        self.audio_fc = nn.Linear(embed_dim, audio_vocab_size)
        self.register_buffer(
            "pos",
            build_positional_embeddings(
                max(text_max_len, audio_max_len),
                embed_dim,
            ),
        )
        self.register_buffer(
            "attn_mask",
            torch.triu(
                torch.ones(
                    text_max_len + audio_max_len,
                    text_max_len + audio_max_len,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the stacked transformer layers; return output and updated KV cache."""
        new_cache = None
        for layer_index, layer in enumerate(self.transformer):
            layer_cache = None if cache is None else cache[layer_index]
            hidden_states, layer_cache = layer(
                hidden_states,
                layer_cache,
                attn_mask,
                key_padding_mask,
            )
            layer_cache = layer_cache.unsqueeze(0)
            new_cache = (
                layer_cache
                if new_cache is None
                else torch.concat([new_cache, layer_cache])
            )
        return hidden_states, new_cache

    def asr(
        self,
        text: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ASR forward: predict text tokens.

        When audio is provided, concatenates [audio | text] for teacher
        forcing / first decode step. When audio is None, uses cache for the
        next token only (incremental decoding).
        """
        text_len = text.size(1)
        text_hidden = self.text_embed(text) + self.pos[:, :text_len]

        if audio is not None:
            audio_len = audio.size(1)
            audio_hidden = self.audio_embed(audio) + self.pos[:, :audio_len]
            hidden_states = torch.concat([audio_hidden, text_hidden], dim=1)
            attn_mask = self.attn_mask[
                : audio_len + text_len, : audio_len + text_len
            ].clone()
            # Allow full attention within the audio prefix (plus first text step).
            attn_mask[: audio_len + 1, : audio_len + 1] = False
        else:
            hidden_states = text_hidden[:, -1:]
            attn_mask = None

        hidden_states, cache = self.forward(
            hidden_states,
            cache,
            attn_mask,
            key_padding_mask,
        )
        logits = self.text_fc(self.norm_fc(hidden_states[:, -text_len:]))
        return logits, cache

    def tts(
        self,
        audio: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TTS forward: predict audio codes.

        When text is provided, concatenates [text | audio]. When text is None,
        uses cache for incremental decoding of the next audio token.
        """
        audio_len = audio.size(1)
        audio_hidden = self.audio_embed(audio) + self.pos[:, :audio_len]

        if text is not None:
            text_len = text.size(1)
            text_hidden = self.text_embed(text) + self.pos[:, :text_len]
            hidden_states = torch.concat([text_hidden, audio_hidden], dim=1)
            attn_mask = self.attn_mask[
                : text_len + audio_len, : text_len + audio_len
            ].clone()
            # Allow full attention within the text prefix (plus first audio step).
            attn_mask[: text_len + 1, : text_len + 1] = False
        else:
            hidden_states = audio_hidden[:, -1:]
            attn_mask = None

        hidden_states, cache = self.forward(
            hidden_states,
            cache,
            attn_mask,
            key_padding_mask,
        )
        logits = self.audio_fc(self.norm_fc(hidden_states[:, -audio_len:]))
        return logits, cache


class TextAudioDataset(torch.utils.data.Dataset):
    """PyTorch Dataset over a dataframe of (text, audio-code) rows."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_tokenizer: TextTokenizer,
        audio_tokenizer: AudioTokenizer,
    ):
        self.dataframe = dataframe
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        text = self.dataframe.iloc[index, 0]
        audio_codes = self.dataframe.iloc[index, 1]
        return (
            self.text_tokenizer.encode(text, training=True),
            self.audio_tokenizer.encode(audio_codes, training=True),
        )
