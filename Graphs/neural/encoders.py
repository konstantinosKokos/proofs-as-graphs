from transformers import RobertaModel, BertModel
from torch.nn import LSTM
from ..data.tokenizer import Tokenizer
from .embedding import from_table
from ..typing import Module, Tensor
from torch import ones_like


class Encoder(Module):
    def __init__(self, tokenizer: Tokenizer, device: str):
        super(Encoder, self).__init__()
        if tokenizer.name in {'bert', 'robert'}:
            self.core = BertWrapper(tokenizer, device)
            self.pad_value = self.core.pad_value
        elif tokenizer.name == 'spacy':
            self.core = LSTMWrapper(tokenizer, device)
            self.pad_value = 0

    def forward(self, x: Tensor) -> Tensor:
        return self.core.forward(x)


class LSTMWrapper(Module):
    def __init__(self, tokenizer: Tokenizer, device: str):
        super(LSTMWrapper, self).__init__()
        self.embedding = from_table(tokenizer.word_tokenizer.get_embedding_table(), False).to(device)
        self.lstm = LSTM(input_size=tokenizer.word_tokenizer.dim, hidden_size=256, bidirectional=True,
                         num_layers=2, batch_first=True, dropout=0.5).to(device)

    def forward(self, word_ids: Tensor) -> Tensor:
        ctx, _ = self.lstm(self.embedding(word_ids))
        return ctx



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

