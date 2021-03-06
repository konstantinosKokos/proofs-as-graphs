from .preprocessing import proofnet_to_graphdata, tokenize_data
from .tokenizer import Tokenizer, load_tokenizer
from ..typing import Dict
from LassyExtraction.aethel import ProofNet
import pickle


def parsable(pn: ProofNet) -> bool:
    try:
        ps, ns = list(zip(*pn.axiom_links))
        assert len(set(ps)) == len(set(ns)) == len(pn.axiom_links)
        return True
    except (ValueError, AssertionError, KeyError):
        return False


def proc_sick(data_file: str = './everything.p', encoder: str = 'spacy'):
    print('Loading tokenizer..')
    tokenizer = load_tokenizer(encoder)

    label_map = {'ENTAILMENT': 0, 'NEUTRAL': 1, 'CONTRADICTION': 2}
    print('Loading file..')
    with open(data_file, 'rb') as f:
        _sents, samples, a_nets, n_nets = pickle.load(f)
    available_nets = [list(filter(parsable, sum(ns, [])))[:1] for ns in zip(n_nets, a_nets)]
    fixed_labels = get_fixed_labels()
    print('Making graphs..')
    available_graphs = [[proofnet_to_graphdata(pn) for pn in sent] for sent in available_nets]
    print('Tokenizing graphs..')
    tokenized = [[tokenize_data(g, tokenizer.atoms_to_ids, tokenizer.words_to_ids) for g in subset]
                 for subset in available_graphs]
    train, dev, test = [], [], []
    label_counts = [0, 0, 0]
    for idx, sent_a, sent_b, _, subset in samples:
        label = fixed_labels[int(idx)]
        graphs_a, graphs_b, label, subset = tokenized[sent_a], tokenized[sent_b], label_map[label], subset.rstrip('\n')
        if not graphs_a or not graphs_b:
            continue
        label_counts[label] += 1
        graph_a, graph_b = graphs_a[0], graphs_b[0]
        add_to = train if subset == 'TRAIN' else dev if subset == 'TRIAL' else test
        add_to.append((graph_a, graph_b, label))
    print(f'Label counts: {label_counts}')
    return train, dev, test


def get_fixed_labels(data_file: str = './SICK_whole_corpus_reannotated.csv') -> Dict[int, str]:
    ret = dict()
    with open(data_file, 'r') as f:
        next(f)
        for line in f:
            tabs = line.split('\t')
            ret[int(tabs[0])] = tabs[3]
    return ret


def save_sick(encoder: str):
    with open(f'Graphs/io/{encoder}/processed_sick.p', 'wb') as f:
        pickle.dump(proc_sick(encoder), f)


def load_sick(encoder: str):
    with open(f'Graphs/io/{encoder}/processed_sick.p', 'rb') as f:
        return pickle.load(f)
