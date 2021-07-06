from transformers import BertTokenizerFast, BertModel
from ..types import Maybe, List, Tuple, Module


class BertWrapper(Module):
    tokenizer: BertTokenizerFast
    model: BertModel

    def __init__(self):
        super(BertWrapper, self).__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained("GroNLP/bert-base-dutch-cased")
        self.model = BertModel.from_pretrained("GroNLP/bert-base-dutch-cased")
        self._cls = [(self.tokenizer.cls_token_id, True)]
        self._sep = [(self.tokenizer.sep_token_id, False)]

    def words_to_ids(self, sentence: List[str]) -> List[Tuple[int, bool]]:
        subwords = [self.tokenizer.encode(w, add_special_tokens=False) for w in sentence]
        ret = [(t, i == len(w) - 1) for w in subwords for i, t in enumerate(w)]
        return self._cls + ret + self._sep

    def ids_to_words(self, ids: List[int]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(ids, False)