from ..data.preprocessing import proofnet_to_graph, tokenize_graph, make_atom_map
# from ..data.tulkens_vectors import vectorize
from ..data.spacy_vectors import vectorize
import pickle


def proc(data_file: str = '../lassy-tlg-extraction/data/train_dev_test_0.4.dev0.p'):
    with open(data_file, 'rb') as df:
        dataset = pickle.load(df)
    graphs = [[proofnet_to_graph(pn) for pn in subset] for subset in dataset]
    atom_map = make_atom_map(sum(graphs, []))
    return [[tokenize_graph(g, atom_map, vectorize) for g in subset] for subset in graphs]



