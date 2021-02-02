from typing import TypeVar, Tuple, List, Iterator, Set, Dict, Union, Callable
from dataclasses import dataclass
from itertools import count
from collections import defaultdict

from LassyExtraction.milltypes import (WordType, FunctorType, DiamondType, BoxType, EmptyType, PolarizedType,
                                       AtomicType, ModalType)
from LassyExtraction.aethel import AxiomLinks, ProofNet

Node_co = TypeVar('Node_co', bound='Node', covariant=True)
T = TypeVar('T')


@dataclass
class Node:
    index: int

    def __hash__(self):
        return self.index


@dataclass
class WNode(Node):
    word: str

    def __hash__(self):
        return self.index


@dataclass
class ANode(Node):
    atom: str
    polarity: bool
    j_idx: int

    def __hash__(self):
        return self.index


@dataclass
class CNode(Node):
    connective: str

    def __hash__(self):
        return self.index


_Graph = Dict[Node, Set[Node]]


def get_nodes(graph: _Graph) -> Set[Node]:
    return set(graph.keys()).union(*graph.values())


def tokenize_graph(graph: _Graph, atom_map: Dict[str, int], word_map: Callable[[str], T]) \
        -> Tuple[List[int], List[T], Tuple[List[int], List[int]]]:

    nodes, edge_index = graph_to_tuple(graph)
    atom_ids = [0 if isinstance(node, WNode) else atom_map[get_atom(node)] for node in nodes]
    word_ids = [word_map(node.word) if isinstance(node, WNode) else 0 for node in nodes]
    return atom_ids, word_ids, edge_index


def get_atom(node: Union[ANode, CNode]) -> str:
    return node.connective if isinstance(node, CNode) else node.atom


def extract_sents(graphs: List[_Graph]) -> Set[str]:
    def extract_sent(graph: _Graph) -> str:
        nodes = sorted(get_nodes(graph), key=lambda node: node.index)
        words = list(map(lambda n: n.word, filter(lambda n: isinstance(n, WNode), nodes)))
        return ' '.join(words)
    return set(map(extract_sent, graphs))


def make_atom_map(graphs: List[_Graph]) -> Dict[str, int]:
    def get_atoms(graph: _Graph) -> Set[str]:
        return set(map(get_atom, filter(lambda node: not isinstance(node, WNode), get_nodes(graph))))
    labels = set.union(*[get_atoms(g) for g in graphs])
    return {label: i for i, label in enumerate(['[PAD]', '[MASK]'] + sorted(labels))}


def proofnet_to_tuple(proofnet: ProofNet) -> Tuple[List[Node], Tuple[List[int], List[int]]]:
    return graph_to_tuple(proofnet_to_graph(proofnet))


def graph_to_tuple(graph: _Graph) -> Tuple[List[Node], Tuple[List[int], List[int]]]:
    nodes = sorted(get_nodes(graph), key=lambda node: node.index)
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    edges = []
    for k in nodes:
        vs = sorted(graph[k], key=lambda v: v.index)
        edges.extend([(node_dict[k.index], node_dict[v.index]) for v in vs])
    edge_index = list(zip(*edges))
    return nodes, edge_index


def proofnet_to_graph(proofnet: ProofNet) -> _Graph:
    # todo: processing as options
    words = proofnet.proof_frame.get_words()
    types = proofnet.proof_frame.get_types()
    words, types = merge_multi_crd(words, types)
    graph = defaultdict(lambda: set())
    add_types(graph, words, types, proofnet.proof_frame.conclusion)
    add_axiom_links(graph, proofnet.axiom_links)
    add_lexical_shortcuts(graph)
    binarize_modalities(graph)
    return graph


def find_by_jidx(graph: _Graph, j_idx: int) -> Node:
    candidates = [node for node in get_nodes(graph) if isinstance(node, ANode) and node.j_idx == j_idx]
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} nodes with {j_idx=}: {candidates}')
    return candidates[0]


def find_targets(graph: _Graph, source: Node) -> Set[Node]:
    return graph[source]


def find_sources(graph: _Graph, target: Node) -> Set[Node]:
    return {node for node in get_nodes(graph) if target in graph[node]}


def add_axiom_links(graph: _Graph, axiom_links: AxiomLinks) -> None:
    for _pos, _neg in axiom_links:
        pos = find_by_jidx(graph, _pos)
        neg = find_by_jidx(graph, _neg)
        sources = find_sources(graph, neg)
        targets = find_targets(graph, neg)
        graph[pos] = graph[pos].union(targets)
        for source in sources:
            graph[source].remove(neg)
            graph[source].add(pos)
        del graph[neg]
    return


def add_lexical_shortcuts(graph: _Graph) -> None:
    wnodes = sorted(filter(lambda node: isinstance(node, WNode), get_nodes(graph)), key=lambda node: node.index)
    for src, tgt in zip(wnodes, wnodes[1:]):
        graph[src].add(tgt)
    return


def binarize_modalities(graph: _Graph) -> None:
    modalities = list(filter(lambda node: isinstance(node, CNode) and node.connective != '→', get_nodes(graph)))

    for modality in modalities:
        targets = {target for target in find_targets(graph, modality)
                   if isinstance(target, CNode) and target.connective == '→'}
        if not targets:
            continue
        targets2 = set.union(*[find_targets(graph, target) for target in targets])
        sources = set.union(*[find_sources(graph, target) for target in targets]) - {modality}

        graph[modality] -= targets
        graph[modality] = graph[modality].union(targets2)
        for target in targets:
            del graph[target]
        for source in sources:
            graph[source] -= targets
            graph[source].add(modality)
    return


def add_types(graph: _Graph, words: List[str], wordtypes: List[WordType], conclusion: PolarizedType) -> None:
    counter = count()
    for word, wordtype in zip(words, wordtypes):
        wordtype = collate_type(wordtype)
        wnodes = [WNode(index=next(counter), word=subword) for subword in word.split()]
        for src, tgt in zip(wnodes, wnodes[1:]):
            graph[src].add(tgt)
        root = add_type(graph, wordtype, counter)
        graph[wnodes[-1]].add(root)
    graph[ANode(next(counter), atom=conclusion.depolarize().type, polarity=False, j_idx=conclusion.index)] = set()
    return


def add_type(graph: _Graph, wordtype: WordType, vargen: Iterator[int]) -> Node:

    def fn(wt: WordType, pol: bool) -> Node:
        if isinstance(wt, PolarizedType):
            return ANode(next(vargen), atom=wt.depolarize().type, polarity=pol, j_idx=wt.index)
        if isinstance(wt, FunctorType):
            node = CNode(next(vargen), '→')
            # graph.add_node(node, depth=depth)
            argnode = fn(wt.argument, not pol)
            resnode = fn(wt.result, pol)
            pos, neg = (resnode, argnode) if pol else (argnode, resnode)
            graph[node].add(pos)
            graph[neg].add(node)
            return node
        if isinstance(wt, DiamondType):
            node = CNode(next(vargen), wt.modality)
            argnode = fn(wt.content, pol)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph[src].add(tgt)
            return node
        if isinstance(wt, BoxType):
            node = CNode(next(vargen), wt.modality)
            argnode = fn(wt.content, pol)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph[src].add(tgt)
            return node
        raise TypeError

    root = fn(wordtype, True)
    return root


def merge_multi_crd(words: List[str], wordtypes: List[WordType]) -> Tuple[List[str], List[WordType]]:
    ret = []
    empties = []
    for word, wordtype in reversed(list(zip(words, wordtypes))):
        if isinstance(wordtype, EmptyType):
            empties.append(word)
        elif empties and (isinstance(wordtype, FunctorType) and isinstance(wordtype.argument, DiamondType)
                          and wordtype.argument.modality == 'cnj'):
            empty = empties.pop()
            ret.append((f'{word} {empty}', wordtype))
        else:
            ret.append((word, wordtype))
    return tuple(zip(*reversed(ret)))


def collate_atom(atom: str) -> str:
    return 'NP' if atom == 'SPEC' else atom


def collate_type(wordtype: WordType) -> WordType:
    if isinstance(wordtype, AtomicType):
        wordtype.type = collate_atom(wordtype.type)
        return wordtype
    elif isinstance(wordtype, FunctorType):
        collate_type(wordtype.argument)
        collate_type(wordtype.result)
        return wordtype
    elif isinstance(wordtype, ModalType):
        collate_type(wordtype.content)
        return wordtype
    raise TypeError(f'Unexpected argument {wordtype} of type {type(wordtype)}')
