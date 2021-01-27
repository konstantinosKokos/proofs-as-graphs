import spacy
from typing import Optional as Maybe

model = spacy.load('nl_core_news_md')


def vectorize(word: str) -> Maybe[int]:
    doc = model(word)
    return doc[0].lex_id if doc[0].has_vector else None
