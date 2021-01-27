import spacy
from typing import Set, Callable
from typing import Optional as Maybe
import numpy as np
from collections import defaultdict

model = spacy.load('nl_core_news_md')


def make_word_map(sents: Set[str]) -> Callable[[str], int]:
    def cdict(word: str) -> int:
        return ret[word]
    docs = model.pipe(sents, disable=['parser', 'tagger', 'ner'])
    ret = defaultdict(lambda: 20000)
    for doc in docs:
        for token in doc:
            ret[token.text] = token.lex_id if token.has_vector else 20000
    return cdict


def vectorize(word: str) -> Maybe[int]:
    doc = model(word)
    return doc[0].lex_id if doc[0].has_vector else 20000


def get_embedding_table() -> np.array:
    vectors = model.vocab.vectors.data
    return np.concatenate([vectors, np.zeros((1, 300))])
