from ..data.preprocessing import proofnet_to_graphdata, tokenize_data, make_atom_map
from ..data.tokenizer import Tokenizer
import pickle
from tqdm import tqdm


def proc_lassy(data_file: str = '../lassy-tlg-extraction/data/train_dev_test_0.4.dev0.p', encoder: str = 'spacy'):
    print('Loading file..')
    with open(data_file, 'rb') as df:
        dataset = pickle.load(df)
    print('Making graphs..')
    graphs = [[proofnet_to_graphdata(pn) for pn in tqdm(subset)] for subset in dataset]
    print('Making atom map..')
    atom_map = make_atom_map(sum(graphs, []))
    print('Building tokenizer..')
    tokenizer = Tokenizer(atom_map, encoder)
    print('Tokenizing graphs..')
    tokenized = [[tokenize_data(g, tokenizer.atoms_to_ids, tokenizer.words_to_ids) for g in tqdm(subset)]
                 for subset in graphs]
    return tokenized, tokenizer


def save_lassy(encoder: str):
    tokenized, tokenizer = proc_lassy()
    with open(f'Graphs/io/{encoder}/processed_lassy.p', 'wb') as f:
        pickle.dump(tokenized, f)
    with open(f'Graphs/io/{encoder}/tokenizer.p', 'wb') as f:
        pickle.dump(tokenizer, f)


def load_lassy(encoder: str):
    with open(f'Graphs/io/{encoder}/processed_lassy.p', 'rb') as f:
        return pickle.load(f)