from ..typing import Dict, List, Tuple
import numpy as np
import pickle
import spacy
from warnings import warn


np.random.seed(42)


class Tokenizer:
    def __init__(self, atom_map: Dict[str, int], word_tokenizer: str):
        assert word_tokenizer in {'bert', 'robert', 'spacy'}
        self.atom_map = atom_map
        self.inv_atom_map = {v: k for k, v in self.atom_map.items()}
        self.name = word_tokenizer
        if word_tokenizer in {'bert', 'robert'}:
            self.word_tokenizer = BertTWrapper(word_tokenizer)
        elif word_tokenizer == 'spacy':
            self.word_tokenizer = SpacyTWrapper(spacy.load('nl_core_news_lg'))

    def atoms_to_ids(self, atoms: List[str]) -> List[int]:
        try:
            return [self.atom_map[a] for a in atoms]
        except KeyError:
            unks = filter(lambda a: a not in self.atom_map.keys(), atoms)
            for unk in unks:
                warn(f'Adding new atom {unk} to the atom map')
                self.atom_map[unk] = len(self.atom_map)
            return self.atoms_to_ids(atoms)

    def ids_to_atoms(self, ids: List[int]) -> List[str]:
        return [self.inv_atom_map[i] for i in ids]

    def words_to_ids(self, words: List[Tuple[str, bool]]) -> List[Tuple[int, bool]]:
        return self.word_tokenizer.words_to_ids(words)

    def ids_to_words(self, ids: List[int]) -> List[str]:
        return self.word_tokenizer.ids_to_words(ids)


def load_tokenizer(which: str) -> Tokenizer:
    with open(f'./Graphs/io/{which}/tokenizer.p', 'rb') as f:
        return pickle.load(f)


class BertTWrapper:
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
        return self._cls + [(t, i == 0) for w in subwords for i, t in enumerate(w)] + self._sep

    def ids_to_words(self, ids: List[int]) -> List[str]:
        return self.core.convert_ids_to_tokens(ids, True)


class SpacyTWrapper:
    def __init__(self, model: spacy.language.Language):
        self.model = model
        self.unk, self.dim = self.model.vocab.vectors.data.shape
        self.pad_value = self.unk + 1

    def words_to_ids(self, words: List[Tuple[str, bool]]) -> List[Tuple[int, bool]]:
        words, masks = list(zip(*words))
        docs = self.model.pipe(words, disable=['parser', 'tagger', 'ner'])
        return [(token.lex_id if token.has_vector else self.unk, i == 0 and not mask)
                for tokens, mask in zip(docs, masks) for i, token in enumerate(tokens)]

    def ids_to_words(self, ids: List[int]):
        raise NotImplementedError

    def get_embedding_table(self) -> np.array:
        vectors = self.model.vocab.vectors.data
        return np.concatenate([vectors, np.random.rand(1, 300), np.zeros((1, 300))])