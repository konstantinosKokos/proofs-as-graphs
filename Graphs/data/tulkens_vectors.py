from gensim.models import KeyedVectors
from numpy import array, zeros

model = KeyedVectors.load_word2vec_format('./cow-320/320/cow-320.txt')


def vectorize(word: str) -> array:
    try:
        return model[word]
    except KeyError:
        return zeros(320, dtype=float)
