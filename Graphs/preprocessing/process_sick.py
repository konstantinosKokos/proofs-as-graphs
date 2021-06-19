from .graphdata import proofnet_to_graphdata, get_vocab, GraphData, tokenize_graph
from ..types import Tuple, Dict, List, Callable, Maybe

import pickle


def proc_sick(data_file: str, word_tokenizer: Maybe[Callable[[List[str]], List[Tuple[int, bool]]]] = None) \
        -> Tuple[Dict[int, Tuple[str, GraphData]],
                 List[Tuple[int, int, int, str, str]],
                 Dict[str, int],
                 Dict[str, int]]:
    with open(data_file, 'rb') as f:
        sents, nets, samples = pickle.load(f)
    samples = [(s_idx, h_idx, p_idx, label, subset) for (s_idx, h_idx, p_idx, label, subset) in samples
               if nets[h_idx] is not None and nets[p_idx] is not None]
    sents = {i: (sents[i], proofnet_to_graphdata(nets[i]))
             for i in sorted(set(map(lambda s: s[1], samples)).union(set(map(lambda s: s[2], samples))))}
    sents = {k: v for k, v in sents.items() if v[1] is not None}
    samples = [s for s in samples if s[1] in sents.keys() and s[2] in sents.keys()]
    node_vocab, edge_vocab = get_vocab([graph for _, graph in sents.values()])
    if word_tokenizer is not None:
        sents = tokenize_sick(sents, node_vocab, edge_vocab, word_tokenizer)
    return sents, samples, node_vocab, edge_vocab


def tokenize_sick(sents: Dict[int, Tuple[str, GraphData]], node_vocab: Dict[str, int], edge_vocab: Dict[str, int],
                  word_tokenizer: Callable[[List[str]], List[Tuple[int, bool]]]) -> Dict[int, Tuple[str, GraphData]]:
    return {i: (sent, tokenize_graph(graph, node_vocab, edge_vocab, word_tokenizer))
            for i, (sent, graph) in sents.items()}
