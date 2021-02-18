from transformers import RobertaModel, BertModel
from torch.nn import LSTM
from torch.nn.functional import dropout
from ..data.tokenizer import Tokenizer, BertTWrapper, SpacyTWrapper
from .embedding import from_table
from ..typing import Module, Tensor, Maybe
from torch import ones_like


class Encoder(Module):
    def __init__(self, tokenizer: Tokenizer, device: str, hidden_size: Maybe[int] = None):
        super(Encoder, self).__init__()
        if tokenizer.name in {'bert', 'robert'}:
            assert isinstance(tokenizer.word_tokenizer, BertTWrapper)
            self.core = BertWrapper(tokenizer, device)
            self.pad_value = self.core.pad_value
        elif tokenizer.name == 'spacy':
            assert isinstance(tokenizer.word_tokenizer, SpacyTWrapper)
            self.pad_value = tokenizer.word_tokenizer.pad_value
            self.core = LSTMWrapper(tokenizer, device, hidden_size, self.pad_value)
        else:
            raise ValueError
        print(f'Initialized a {type(self.core)} encoder')

    def forward(self, x: Tensor) -> Tensor:
        return self.core.forward(x)


class LSTMWrapper(Module):
    def __init__(self, tokenizer: Tokenizer, device: str, hidden_size: Maybe[int], pad_value: int):
        super(LSTMWrapper, self).__init__()
        hidden_size = 256 if hidden_size is None else hidden_size
        self.embedding = from_table(tokenizer.word_tokenizer.get_embedding_table(), False, pad_value).to(device)
        self.lstm = LSTM(input_size=tokenizer.word_tokenizer.dim, hidden_size=hidden_size, bidirectional=True,
                         num_layers=1, batch_first=True).to(device)

    def forward(self, word_ids: Tensor) -> Tensor:
        ctx, _ = self.lstm(self.embedding(word_ids))
        return sum(ctx.chunk(2, dim=-1))


class BertWrapper(Module):
    def __init__(self, tokenizer: Tokenizer, device: str):
        super(BertWrapper, self).__init__()
        if tokenizer.name == 'robert':
            self.core = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base").to(device)
        elif tokenizer.name == 'bert':
            self.core = BertModel.from_pretrained("wietsedv/bert-base-dutch-cased").to(device)
        self.pad_value = tokenizer.word_tokenizer.core.pad_token_id
        self.device = device

    def make_mask(self, inps: Tensor, padding_id: int) -> Tensor:
        mask = ones_like(inps)
        mask[inps == padding_id] = 0
        return mask.to(self.device)

    def make_word_mask(self, lexical_ids: Tensor) -> Tensor:
        return self.make_mask(lexical_ids, self.pad_value).to(self.device)

    def forward(self, word_ids: Tensor) -> Tensor:
        out = self.core(word_ids.to(self.device), attention_mask=self.make_word_mask(word_ids))['last_hidden_state']
        return out
