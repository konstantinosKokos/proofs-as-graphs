from ..data.preprocessing import proofnet_to_graph, tokenize_graph, make_atom_map, extract_sents, Graph
from ..data.process_lassy import load_lassy

from LassyExtraction.aethel import ProofNet, Term

from Graphs.typing import Dict, Maybe
import pickle
from tqdm import tqdm

print('Loading atom map..')
_, atom_map, _ = load_lassy()


def parsable(pn: ProofNet) -> bool:
    try:
        ps, ns = list(zip(*pn.axiom_links))
        assert len(set(ps)) == len(set(ns)) == len(pn.axiom_links)
        return True
    except (ValueError, AssertionError, KeyError):
        return False


def proc_sick(data_file: str = './everything.p'):
    from ..data.spacy_vectors import make_word_map

    label_map = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}

    print('Loading file..')
    with open(data_file, 'rb') as f:
        sents, samples, a_nets, n_nets = pickle.load(f)
    available_nets = [list(filter(parsable, sum(ns, [])))[:1] for ns in zip(n_nets, a_nets)]
    print('Making graphs..')
    available_graphs = [[proofnet_to_graph(pn) for pn in sent] for sent in available_nets]
    print('Extracting sents..')
    sents = extract_sents(sum(available_graphs, []))
    print('Building vocab..')
    word_map = make_word_map(sents)
    print('Tokenizing graphs..')
    tokenized = [[tokenize_graph(g, atom_map, word_map) for g in subset] for subset in available_graphs]
    train, dev, test = [], [], []
    for _, sent_a, sent_b, label, subset in samples:
        graphs_a, graphs_b, label, subset = tokenized[sent_a], tokenized[sent_b], label_map[label], subset.rstrip('\n')
        if not graphs_a or not graphs_b:
            continue
        graph_a, graph_b = graphs_a[0], graphs_b[0]
        add_to = train if subset == 'TRAIN' else dev if subset == 'TRIAL' else test
        add_to.append((graph_a, graph_b, label))
    return train, dev, test


def load_sick(proc_file: str = './processed_sick.p'):
    with open(proc_file, 'rb') as f:
        return pickle.load(f)
