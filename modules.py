from typing import Optional, Union, List, Callable
from utils import get_vocab
import torch.nn as nn
import pandas as pd
import torch
import math
import ast


class TextTokenizer:
    def __init__(
        self,
        max_len: int,
    ):
        self.max_len = max_len
        self.vocab = get_vocab()
        self.pad_tok = self.vocab["<pad>"]
        self.sos_tok = self.vocab["<sos>"]
        self.eos_tok = self.vocab["<eos>"]
        self.unk_tok = self.vocab["<unk>"]
        self.vocab_size = len(self.vocab.items())
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def encode(
        self,
        x: Optional[str] = None,
        training: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if x == None:
            return torch.tensor([self.sos_tok])
        x = [self.vocab.get(i, self.unk_tok) for i in x.lower()]
        x = torch.tensor(
            [self.sos_tok] + x[: self.max_len - 2] + [self.eos_tok],
        )
        if training:
            x = torch.concat(
                (
                    x,
                    torch.full(
                        (self.max_len - x.size(0),),
                        self.pad_tok,
                    ),
                )
            )
            return x, (x) == self.pad_tok
        return x

    def decode(self, x: torch.Tensor) -> str:
        return "".join(
            [
                self.inv_vocab.get(i)
                for i in x.tolist()
                if i
                not in [
                    self.pad_tok,
                    self.sos_tok,
                    self.eos_tok,
                    self.unk_tok,
                ]
            ]
        )


class AudioTokenizer:
    def __init__(
        self,
        max_len: int,
    ):
        self.max_len = max_len
        self.pad_tok = 4096
        self.sos_tok = 4097
        self.eos_tok = 4098
        self.vocab_size = 4099

    def encode(
        self,
        x: Optional[Union[torch.Tensor, str]] = None,
        training: bool = False,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if x == None:
            return torch.tensor([self.sos_tok])
        if type(x) == str:
            x = torch.tensor(ast.literal_eval(x))
        x = torch.concat(
            [
                torch.tensor([self.sos_tok]),
                x[: self.max_len - 2],
                torch.tensor([self.eos_tok]),
            ]
        )
        if training:
            x = torch.concat(
                [
                    x,
                    torch.full(
                        (self.max_len - x.size(0),),
                        self.pad_tok,
                    ),
                ]
            )
            return x, (x) == self.pad_tok
        return x


def get_pos_embed(
    max_len: int,
    dim_embed: int,
) -> torch.Tensor:
    pos = torch.arange(0, max_len).unsqueeze(1)
    div = torch.arange(0, dim_embed, 2)
    div = torch.exp(div * -(math.log(10000) / dim_embed))
    embed = torch.zeros(max_len, dim_embed)
    embed[:, 0::2] = torch.sin(pos * div)
    embed[:, 1::2] = torch.cos(pos * div)
    return embed.unsqueeze(0)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.relu,
        batch_first: bool = True,
    ):
        super(TransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            dim_embed,
            nheads,
            dropout,
            batch_first=batch_first,
        )
        self.activation = activation
        self.self_attn_dropout = nn.Dropout(dropout)
        self.self_attn_norm = nn.LayerNorm(dim_embed)
        self.feedforward_dropout1 = nn.Dropout(dropout)
        self.feedforward_dropout2 = nn.Dropout(dropout)
        self.feedforward_norm = nn.LayerNorm(dim_embed)
        self.feedforward_linear1 = nn.Linear(dim_embed, dim_feedforward)
        self.feedforward_linear2 = nn.Linear(dim_feedforward, dim_embed)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> List[torch.Tensor]:
        query = self.self_attn_norm(x)
        key = (
            query
            if cache == None
            else torch.concat(
                [cache, query],
                dim=1,
            )
        )
        x = x + self.self_attn_block(
            query=query,
            key=key,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )
        x = x + self.feedforward_block(self.feedforward_norm(x))
        return x, key

    def self_attn_block(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        return self.self_attn_dropout(
            self.self_attn(
                query=query,
                key=key,
                value=key,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
                is_causal=is_causal,
            )[0]
        )

    def feedforward_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.feedforward_dropout2(
            self.feedforward_linear2(
                self.feedforward_dropout1(self.activation(self.feedforward_linear1(x)))
            )
        )


class Model(nn.Module):
    def __init__(
        self,
        text_vocab_size: int,
        audio_vocab_size: int,
        dim_embed: int,
        nheads: int,
        nlayers: int,
        dim_feedforward: int,
        text_max_len: int,
        audio_max_len: int,
    ):
        super(Model, self).__init__()
        self.text_embed = nn.Embedding(text_vocab_size, dim_embed)
        self.audio_embed = nn.Embedding(audio_vocab_size, dim_embed)
        self.transformer = nn.ModuleList(
            [
                TransformerLayer(
                    dim_embed,
                    nheads,
                    dim_feedforward,
                )
                for _ in range(nlayers)
            ]
        )
        self.norm_fc = nn.LayerNorm(dim_embed)
        self.text_fc = nn.Linear(dim_embed, text_vocab_size)
        self.audio_fc = nn.Linear(dim_embed, audio_vocab_size)
        self.register_buffer(
            "pos",
            get_pos_embed(
                max(text_max_len, audio_max_len),
                dim_embed,
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
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        ncache = None
        for idx, layer in enumerate(self.transformer):
            lcache = None if cache == None else cache[idx]
            x, lcache = layer(
                x,
                lcache,
                attn_mask,
                key_padding_mask,
            )
            lcache = lcache.unsqueeze(0)
            ncache = lcache if ncache == None else torch.concat([ncache, lcache])
        return x, ncache

    def asr(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        xlen = x.size(1)
        x = self.text_embed(x)
        x = x + self.pos[:, :xlen]
        if not y == None:
            ylen = y.size(1)
            y = self.audio_embed(y)
            y = y + self.pos[:, :ylen]
            x = torch.concat([y, x], dim=1)
            attn_mask = self.attn_mask[: ylen + xlen, : ylen + xlen].clone()
            attn_mask[: ylen + 1, : ylen + 1] = False
        else:
            x = x[:, -1:]
            attn_mask = None
        x, cache = self.forward(
            x,
            cache,
            attn_mask,
            key_padding_mask,
        )
        x = self.text_fc(self.norm_fc(x[:, -xlen:]))
        return x, cache

    def tts(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        cache: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        xlen = x.size(1)
        x = self.audio_embed(x)
        x = x + self.pos[:, :xlen]
        if not y == None:
            ylen = y.size(1)
            y = self.text_embed(y)
            y = y + self.pos[:, :ylen]
            x = torch.concat([y, x], dim=1)
            attn_mask = self.attn_mask[: ylen + xlen, : ylen + xlen].clone()
            attn_mask[: ylen + 1, : ylen + 1] = False
        else:
            x = x[:, -1:]
            attn_mask = None
        x, cache = self.forward(
            x,
            cache,
            attn_mask,
            key_padding_mask,
        )
        x = self.audio_fc(self.norm_fc(x[:, -xlen:]))
        return x, cache


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_tokenizer: TextTokenizer,
        audio_tokenizer: AudioTokenizer,
    ):
        self.dataframe = dataframe
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        return self.text_tokenizer.encode(
            self.dataframe.iloc[idx, 0],
            training=True,
        ), self.audio_tokenizer.encode(
            self.dataframe.iloc[idx, 1],
            training=True,
        )
