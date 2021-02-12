from ..typing import Dict, List, Tuple
import pickle
import spacy


class Tokenizer:
    def __init__(self, atom_map: Dict[str, int], word_tokenizer: str):
        assert word_tokenizer in {'bert', 'robert'}
        self.atom_map = atom_map
        self.inv_atom_map = {v: k for k, v in self.atom_map.items()}
        if word_tokenizer in {'bert', 'robert'}:
            self.word_tokenizer = BertWrapper(word_tokenizer)
        elif word_tokenizer == 'spacy':
            self.word_tokenizer = SpacyWrapper(spacy.load('nl_core_news_lg'))

    def atoms_to_ids(self, atoms: List[str]) -> List[int]:
        return [self.atom_map[a] for a in atoms]

    def ids_to_atms(self, ids: List[int]) -> List[str]:
        return [self.inv_atom_map[i] for i in ids]

    def words_to_ids(self, words: List[str]) -> List[Tuple[int, bool]]:
        return self.word_tokenizer.words_to_ids(words)

    def ids_to_words(self, ids: List[int]) -> List[str]:
        return self.word_tokenizer.ids_to_words(ids)


def load_tokenizer(path: str = './tokenizer.p') -> Tokenizer:
    with open(path, 'rb') as f:
        return pickle.load(f)


class BertWrapper:
    def __init__(self, model: str):
        if model == 'bert':
            from transformers import BertTokenizer
            self.core = BertTokenizer.from_pretrained('wietsedv/bert-base-dutch-cased')
        else:
            from transformers import RobertaTokenizer
            self.core = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self._cls = [(self.core.cls_token_id, False)]
        self._sep = [(self.core.sep_token_id, False)]

    def words_to_ids(self, words: List[str]) -> List[Tuple[int, bool]]:
        subwords = [self.core.encode(w, add_special_tokens=False) for w in words]
        return self._cls + [(t, i == (len(w) - 1)) for w in subwords for i, t in enumerate(w)] + self._sep

    def ids_to_words(self, ids: List[int]) -> List[str]:
        return self.core.convert_ids_to_tokens(ids, True)


class SpacyWrapper:
    def __init__(self, model: spacy.language.Language):
        self.model = model
        self.unk, self.dim = self.model.vocab.vectors.data.shape

    def words_to_ids(self, words: List[str]) -> List[Tuple[int, bool]]:
        docs = self.model.pipe(words, disable=['parser', 'tagger', 'ner'])
        return [(token.lex_id if token.has_vector else self.unk, i == (len(tokens) - 1))
                for tokens in docs for i, token in enumerate(tokens)]

    def ids_to_words(self, ids: List[int]):
        raise NotImplementedError
