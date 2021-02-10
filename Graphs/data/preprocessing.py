from ..typing import (Tuple, List, Iterator, Set, Dict, Callable, Node, ANode, CNode, WNode, Graph, GraphData)
from collections import defaultdict
from itertools import count

from LassyExtraction.milltypes import (WordType, FunctorType, DiamondType, BoxType, EmptyType, PolarizedType,
                                       AtomicType, ModalType)
from LassyExtraction.aethel import AxiomLinks, ProofNet


def get_nodes(graph: Graph) -> Set[Node]:
    return set(filter(lambda k: k is not None, graph.keys())).union(*graph.values())


def tokenize_data(untokenized: GraphData, atom_map: Dict[str, int],
                  word_map: Callable[[List[str]], List[Tuple[int, bool]]]) -> GraphData:
    atom_ids = [atom_map[atom] for atom in untokenized.nodes]
    ids_and_starts = word_map(untokenized.words)
    return GraphData(words=ids_and_starts, nodes=atom_ids, edges=untokenized.edges,
                     conclusion=untokenized.conclusion, roots=untokenized.roots)


def make_atom_map(graphs: List[GraphData]) -> Dict[str, int]:
    labels = set.union(*[set(g.nodes) for g in graphs])
    return {label: i for i, label in enumerate(['[PAD]', '[MASK]'] + sorted(labels))}


def proofnet_to_graphdata(proofnet: ProofNet) -> GraphData:
    # Create graph structure
    words = proofnet.proof_frame.get_words()
    types = proofnet.proof_frame.get_types()
    graph = defaultdict(lambda: set())
    lex_ancors, cnode = add_types(graph, words, types, proofnet.proof_frame.conclusion)
    cnode = add_axiom_links(graph, cnode, proofnet.axiom_links)
    binarize_modalities(graph)
    remove_lex(graph, lex_ancors)

    # Now convert to graphdata
    nodes = sorted(get_nodes(graph), key=lambda node: node.index)
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    srcs, tgts = [], []
    for k in nodes:
        vs = sorted(graph[k], key=lambda v: v.index)
        srcs.extend([node_dict[k.index] for _ in vs])
        tgts.extend([node_dict[v.index] for v in vs])
    return GraphData(nodes=[node.label for node in nodes], edges=(srcs, tgts), words=words,
                     roots=sorted([node_dict[root.index] for root in graph[None]]), conclusion=cnode.index)


def find_by_jidx(graph: Graph, j_idx: int) -> Node:
    candidates = [node for node in get_nodes(graph) if isinstance(node, ANode) and node.j_idx == j_idx]
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} nodes with {j_idx=}: {candidates}')
    return candidates[0]


def find_targets(graph: Graph, source: Node) -> Set[Node]:
    return graph[source]


def find_sources(graph: Graph, target: Node) -> Set[Node]:
    return {node for node in get_nodes(graph) if target in graph[node]}


def add_axiom_links(graph: Graph, cnode: ANode, axiom_links: AxiomLinks) -> ANode:
    for _pos, _neg in axiom_links:
        pos = find_by_jidx(graph, _pos)
        neg = find_by_jidx(graph, _neg)
        if neg == cnode:
            cnode = pos
        sources = find_sources(graph, neg)
        targets = find_targets(graph, neg)
        graph[pos] = graph[pos].union(targets)
        for source in sources:
            graph[source].remove(neg)
            graph[source].add(pos)
        del graph[neg]
    return cnode


def binarize_modalities(graph: Graph) -> None:
    modalities = list(filter(lambda node: isinstance(node, CNode) and node.label != '→', get_nodes(graph)))

    for modality in modalities:
        targets = {target for target in find_targets(graph, modality)
                   if isinstance(target, CNode) and target.label == '→'}
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


def add_types(graph: Graph, words: List[str], wordtypes: List[WordType], conclusion: PolarizedType) \
        -> Tuple[List[WNode], ANode]:
    counter = count()
    wordtypes = list(map(collate_type, wordtypes))
    roots = []
    for word, wordtype in zip(words, wordtypes):
        wnode = WNode(next(counter), word)
        roots.append(wnode)
        graph[wnode].add(add_type(graph, wordtype, counter))
    conc = ANode(next(counter), label=conclusion.depolarize().type, polarity=False, j_idx=conclusion.index)
    graph[conc] = set()
    return roots, conc


def add_type(graph: Graph, wordtype: WordType, vargen: Iterator[int]) -> Node:

    def fn(wt: WordType, pol: bool) -> Node:
        if isinstance(wt, PolarizedType):
            return ANode(next(vargen), label=wt.depolarize().type, polarity=pol, j_idx=wt.index)
        if isinstance(wt, FunctorType):
            node = CNode(next(vargen), '→')
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
        if isinstance(wt, EmptyType):
            return ANode(next(vargen), label=wt.type, polarity=pol, j_idx=-1)

    root = fn(wordtype, True)
    return root


def remove_lex(graph: Graph, roots: List[Node]):
    empties = []
    for root in reversed(roots):
        _tgt = list(graph[root])
        assert len(_tgt) == 1
        tgt = _tgt[0]
        if tgt.label == '_':
            empties.append(tgt)
        if tgt.label == 'cnj' and empties:
            empty = empties.pop()
            tgts = graph[tgt]
            assert len(tgts) == 1
            graph[empty] = graph[empty].union(tgts)
        graph[None] = graph[None].union(graph[root])
        del graph[root]


# def merge_multi_crd(words: List[str], wordtypes: List[WordType]) -> Tuple[List[str], List[WordType]]:
#     ret = []
#     empties = []
#     for word, wordtype in reversed(list(zip(words, wordtypes))):
#         if isinstance(wordtype, EmptyType):
#             empties.append(word)
#         elif empties and (isinstance(wordtype, FunctorType) and isinstance(wordtype.argument, DiamondType)
#                           and wordtype.argument.modality == 'cnj'):
#             empty = empties.pop()
#             ret.append((f'{word} {empty}', wordtype))
#         else:
#             ret.append((word, wordtype))
#     return tuple(zip(*reversed(ret)))


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


def visualize(graph: Graph):
    import graphviz as gv

    dg = gv.Digraph()

    for s in graph.keys():
        for t in graph[s]:
            dg.node(str(s.index) if s is not None else 'SRC', label=str(s))
            dg.node(str(t.index), label=str(t))

            dg.edge(str(s.index) if s is not None else 'SRC', str(t.index))
    dg.render(view=True)