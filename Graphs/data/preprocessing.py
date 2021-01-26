from typing import TypeVar, Tuple, List, Iterator, Set, Dict, Union
from dataclasses import dataclass
from itertools import count

import networkx as nx

from LassyExtraction.milltypes import WordType, FunctorType, DiamondType, BoxType, EmptyType, PolarizedType
from LassyExtraction.aethel import AxiomLinks, ProofNet

T_co = TypeVar('T_co', bound='Node', covariant=True)


@dataclass(unsafe_hash=True)
class Node:
    index: int

    def __str__(self):
        return str(self.index)


@dataclass(unsafe_hash=True)
class WNode(Node):
    word: str

    def __str__(self):
        return self.word

    def __repr__(self):
        return str(self)


@dataclass(unsafe_hash=True)
class ANode(Node):
    atom: str
    polarity: bool
    j_idx: int

    def __str__(self):
        return self.atom

    def __repr__(self):
        return str(self)


@dataclass(unsafe_hash=True)
class CNode(Node):
    connective: str

    def __str__(self):
        return self.connective

    def __repr__(self):
        return str(self)


def tokenize_graph(graph: nx.DiGraph, atom_map: Dict[str, int], word_map: Dict[str, int]):
    nodes, edge_index = graph_to_tuple(graph)
    atom_ids = [0 if isinstance(node, WNode) else atom_map[get_atom(node)] for node in nodes]
    word_ids = [word_map[node.word] if isinstance(node, WNode) else 0 for node in nodes]
    return atom_ids, word_ids, edge_index


def get_atom(node: Union[ANode, CNode]) -> str:
    return node.connective if isinstance(node, CNode) else node.atom


def make_atom_map(graphs: List[nx.DiGraph]) -> Dict[str, int]:
    def get_atoms(graph: nx.DiGraph) -> Set[str]:
        return set(map(get_atom, filter(lambda node: not isinstance(node, WNode), graph.nodes)))
    labels = set.union(*[get_atoms(g) for g in graphs])
    return {label: i + 1 for i, label in enumerate(sorted(labels))}


def graph_to_tuple(graph: nx.DiGraph):
    nodes = sorted(graph.nodes, key=lambda node: node.index)
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    edges = sorted(graph.edges, key=lambda edge: (edge[0].index, edge[1].index))
    edge_index = list(zip(*[(node_dict[edge[0].index], node_dict[edge[1].index]) for edge in edges]))
    return nodes, edge_index


def proofnet_to_graph(proofnet: ProofNet) -> nx.DiGraph:
    # todo: processing as options
    words = proofnet.proof_frame.get_words()
    types = proofnet.proof_frame.get_types()
    words, types = merge_multi_crd(words, types)
    graph = add_axiom_links(types_to_graph(words, types, proofnet.proof_frame.conclusion), proofnet.axiom_links)
    graph = add_lexical_shortcuts(graph)
    graph = binarize_modalities(graph)
    return graph


def find_by_jidx(graph: nx.DiGraph, j_idx: int) -> Node:
    candidates = [node for node in graph.nodes if isinstance(node, ANode) and node.j_idx == j_idx]
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} nodes with {j_idx=}: {candidates}')
    return candidates[0]


def find_targets(graph: nx.DiGraph, index: int) -> Set[Node]:
    return {node for node in graph.nodes if any(map(lambda edge: edge[0].index == index and edge[1] == node,
                                                graph.edges))}


def find_sources(graph: nx.DiGraph, index: int) -> Set[Node]:
    return {node for node in graph.nodes if any(map(lambda edge: edge[0] == node and edge[1].index == index,
                                                graph.edges))}


def add_axiom_links(graph: nx.DiGraph, axiom_links: AxiomLinks) -> nx.DiGraph:
    for _pos, _neg in axiom_links:
        pos = find_by_jidx(graph, _pos)
        neg = find_by_jidx(graph, _neg)
        sources = find_sources(graph, neg.index)
        targets = find_targets(graph, neg.index)
        graph.remove_edges_from([(neg, target) for target in targets])
        graph.remove_edges_from([(source, neg) for source in sources])
        graph.add_edges_from([(source, pos) for source in sources])
        graph.add_edges_from([(pos, target) for target in targets])
        graph.remove_node(neg)
    return graph


def add_lexical_shortcuts(graph: nx.DiGraph) -> nx.DiGraph:
    wnodes = sorted(filter(lambda node: isinstance(node, WNode), graph.nodes), key=lambda node: node.index)
    graph.add_edges_from(zip(wnodes, wnodes[1:]))
    return graph


def binarize_modalities(graph: nx.DiGraph) -> nx.DiGraph:
    modalities = list(filter(lambda node: isinstance(node, CNode) and node.connective != '→', graph.nodes))

    for modality in modalities:
        targets = {target for target in find_targets(graph, modality.index)
                   if isinstance(target, CNode) and target.connective == '→'}
        if not targets:
            continue
        targets2 = set.union(*[find_targets(graph, target.index) for target in targets])
        sources = set.union(*[find_sources(graph, target.index) for target in targets]) - {modality}

        # to remove
        mod_to_t1s = [(modality, target) for target in targets]
        t1s_to_t2s = [(target, target2) for target in targets for target2 in targets2]
        source_to_t1s = [(source, target) for source in sources for target in targets]
        # to add
        mod_to_t2s = [(modality, target) for target in targets2]
        source_to_mod = [(source, modality) for source in sources]

        graph.remove_edges_from(mod_to_t1s + t1s_to_t2s + source_to_t1s)
        graph.add_edges_from(mod_to_t2s + source_to_mod)
        graph.remove_nodes_from(targets)
    return graph


def types_to_graph(words: List[str], wordtypes: List[WordType], conclusion: PolarizedType) -> nx.DiGraph:

    counter = count()
    graph = nx.DiGraph()
    for word, wordtype in zip(words, wordtypes):
        wnodes = [WNode(index=next(counter), word=subword) for subword in word.split()]
        graph.add_nodes_from([(wnode, {'depth': -1}) for wnode in wnodes])
        graph.add_edges_from(list(zip(wnodes, wnodes[1:])))
        type_graph, root = type_to_graph(wordtype, counter)
        graph = nx.union(graph, type_graph)
        graph.add_edge(wnodes[-1], root)
    graph.add_node(ANode(next(counter), atom=conclusion.depolarize().type, polarity=False, j_idx=conclusion.index),
                   depth=0)
    return graph


def type_to_graph(wordtype: WordType, vargen: Iterator[int]) -> Tuple[nx.DiGraph, Node]:
    graph = nx.DiGraph()

    def fn(wt: WordType, pol: bool, depth: int) -> Node:
        if isinstance(wt, PolarizedType):
            node = ANode(next(vargen), atom=wt.depolarize().type, polarity=pol, j_idx=wt.index)
            graph.add_node(node, depth=depth)
            return node
        if isinstance(wt, FunctorType):
            node = CNode(next(vargen), '→')
            graph.add_node(node, depth=depth)
            argnode = fn(wt.argument, not pol, depth+1)
            resnode = fn(wt.result, pol, depth+1)
            pos, neg = (resnode, argnode) if pol else (argnode, resnode)
            graph.add_edges_from([(node, pos), (neg, node)])
            return node
        if isinstance(wt, DiamondType):
            node = CNode(next(vargen), wt.modality)
            graph.add_node(node, depth=depth)
            argnode = fn(wt.content, pol, depth+1)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph.add_edge(src, tgt)
            return node
        if isinstance(wt, BoxType):
            node = CNode(next(vargen), wt.modality)
            graph.add_node(node, depth=depth)
            argnode = fn(wt.content, pol, depth+1)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph.add_edge(src, tgt)
            return node
        raise TypeError

    root = fn(wordtype, True, 0)
    return graph, root


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


#
# def make_atom_map(node_lists: List[List[Node]]) -> Dict[str, int]:
#     nodes = sorted(set(map(lambda n: n.content[0] if isinstance(n, ANode) else n.content,
#                            filter(lambda n: isinstance(n, CNode) or isinstance(n, ANode),
#                                   chain.from_iterable(node_lists)))))
#     return {n: i for i, n in enumerate(nodes)}
