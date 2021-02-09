from ..data.preprocessing import proofnet_to_graph, tokenize_graph, make_atom_map, extract_sents
import pickle
from tqdm import tqdm


def proc_lassy(data_file: str = '../lassy-tlg-extraction/data/train_dev_test_0.4.dev0.p'):
    from ..data.robbert_vectors import encode_words

    print('Loading file..')
    with open(data_file, 'rb') as df:
        dataset = pickle.load(df)
    print('Making graphs..')
    graphs = [[proofnet_to_graph(pn) for pn in tqdm(subset)] for subset in dataset]
    # print('Extracting sents..')
    # sents = extract_sents(sum(graphs, []))
    # print('Building vocab..')
    # word_map = make_word_map(sents)
    print('Making atom map..')
    atom_map = make_atom_map(sum(graphs, []))
    print('Tokenizing graphs..')
    tokenized = [[tokenize_graph(g, atom_map, encode_words) for g in tqdm(subset)] for subset in graphs]
    return tokenized, atom_map  #, get_embedding_table()


def save_lassy(proc_file: str = './processed_lassy.p'):
    with open(proc_file, 'wb') as f:
        pickle.dump(proc_lassy(), f)


def load_lassy(proc_file: str = './processed_lassy.p'):
    with open(proc_file, 'rb') as f:
        return pickle.load(f)