import spacy
from ..typing import Maybe, List, Tuple
import numpy as np

model = spacy.load('nl_core_news_lg')
unk, dim = model.vocab.vectors.data.shape


def word_map(words: List[str]) -> List[Tuple[int, bool]]:
    docs = model.pipe(words, disable=['parser', 'tagger', 'ner'])
    return [(token.lex_id if token.has_vector else unk, i == (len(tokens) - 1))
            for tokens in docs for i, token in enumerate(tokens)]


def vectorize(word: str) -> Maybe[int]:
    doc = model(word)
    return doc[0].lex_id if doc[0].has_vector else unk


def get_embedding_table() -> np.array:
    vectors = model.vocab.vectors.data
    return np.concatenate([vectors, np.zeros((1, dim))])
