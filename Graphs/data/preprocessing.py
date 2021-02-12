from ..typing import (Tuple, List, Iterator, Set, Dict, Callable, Node, ANode, CNode, WNode, Graph, Edge, GraphData)
from collections import defaultdict
from itertools import count

from LassyExtraction.milltypes import (WordType, FunctorType, DiamondType, BoxType, EmptyType, PolarizedType,
                                       AtomicType, ModalType)
from LassyExtraction.aethel import AxiomLinks, ProofNet


lex, arg, res = 'lex', 'arg', 'res'


def tokenize_data(untokenized: GraphData, atom_map: Callable[[List[str]], List[int]],
                  word_map: Callable[[List[str]], List[Tuple[int, bool]]]) -> GraphData:
    edge_map = {lex: 0, arg: 1, res: 2}
    atom_ids = atom_map(untokenized.nodes)
    ids_and_starts = word_map(untokenized.words)
    return GraphData(words=ids_and_starts, nodes=atom_ids, edge_index=untokenized.edge_index,
                     conclusion=untokenized.conclusion, roots=untokenized.roots,
                     edge_attrs=[edge_map[edge] for edge in untokenized.edge_attrs])


def make_atom_map(graphs: List[GraphData]) -> Dict[str, int]:
    labels = set.union(*[set(g.nodes) for g in graphs]).difference({'[PAD]', '[MASK]'})
    return {label: i for i, label in enumerate(['[PAD]', '[MASK]'] + sorted(labels))}


def proofnet_to_graphdata(proofnet: ProofNet) -> GraphData:
    # Create graph structure
    words = proofnet.proof_frame.get_words()
    types = proofnet.proof_frame.get_types()
    graph = defaultdict(lambda: set())
    lex_anchors, cnode = add_types(graph, words, types, proofnet.proof_frame.conclusion)
    cnode = add_axiom_links(graph, cnode, proofnet.axiom_links)
    binarize_modalities(graph)
    merge_multi_crd(graph, lex_anchors)

    # Now convert to graphdata
    nodes = sorted(get_nodes(graph), key=lambda node: node.index)
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    srcs, tgts, attrs = [], [], []
    for k in nodes:
        vs = sorted(graph[k], key=lambda edge: edge.target.index)
        srcs.extend([node_dict[k.index] for _ in graph[k]])
        tgts.extend([node_dict[v.index] for v, _ in vs])
        attrs.extend([e for _, e in vs])

    return GraphData(nodes=['[PAD]' if isinstance(node, WNode) else node.label for node in nodes],
                     edge_index=(srcs, tgts), words=words, edge_attrs=attrs, conclusion=node_dict[cnode.index],
                     roots=sorted([node_dict[lanc.index] for lanc in lex_anchors]))


def get_nodes(graph: Graph) -> Set[Node]:
    return set(graph.keys()).union(*[find_targets(graph, k) for k in graph.keys()])


def find_by_jidx(graph: Graph, j_idx: int) -> Node:
    candidates = [node for node in get_nodes(graph) if isinstance(node, ANode) and node.j_idx == j_idx]
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} nodes with {j_idx=}: {candidates}')
    return candidates[0]


def find_targets(graph: Graph, source: Node) -> Set[Node]:
    return set(map(lambda edge: edge.target, graph[source]))


def find_sources(graph: Graph, target: Node) -> Set[Node]:
    return {node for node in get_nodes(graph) if target in find_targets(graph, node)}


def add_axiom_links(graph: Graph, cnode: ANode, axiom_links: AxiomLinks) -> ANode:
    for _pos, _neg in axiom_links:
        pos = find_by_jidx(graph, _pos)
        neg = find_by_jidx(graph, _neg)
        if neg == cnode:
            cnode = pos
        sources = find_sources(graph, neg)
        targets = graph[neg]
        graph[pos] = graph[pos].union(targets)
        for source in sources:
            graph[source].remove(Edge(neg, arg))
            graph[source].add(Edge(pos, res))
        del graph[neg]
    return cnode


def binarize_modalities(graph: Graph) -> None:
    modalities = list(filter(lambda node: isinstance(node, CNode) and node.label != '→', get_nodes(graph)))

    for modality in modalities:
        targets: Set[Node] = set(filter(lambda node: isinstance(node, CNode) and node.label == '→',
                                        find_targets(graph, modality)))
        if not targets:
            continue
        targets2: Set[Edge] = set.union(*[graph[target] for target in targets])
        sources: Set[Node] = set.union(*[find_sources(graph, target) for target in targets]) - {modality}
        graph[modality] = set(filter(lambda edge: edge.target not in targets, graph[modality]))
        graph[modality] = graph[modality].union(targets2)
        for target in targets:
            del graph[target]
        for source in sources:
            edges = set(map(lambda edge: edge.label, filter(lambda edge: edge.target in targets, graph[source])))
            assert len(edges) == 1
            label = list(edges)[0]
            graph[source] = set(filter(lambda edge: edge.target not in targets, graph[source]))
            graph[source].add(Edge(modality, lex if isinstance(source, WNode) else label))


def add_types(graph: Graph, words: List[str], wordtypes: List[WordType], conclusion: PolarizedType) \
        -> Tuple[List[WNode], ANode]:
    counter = count()
    wordtypes = list(map(collate_type, wordtypes))
    lex_anchors = []
    for word, wordtype in zip(words, wordtypes):
        wnode = WNode(next(counter), word)
        lex_anchors.append(wnode)
        root = add_type(graph, wordtype, counter)
        graph[wnode].add(Edge(root, lex))
    conc = ANode(next(counter), label=conclusion.depolarize().type, polarity=False, j_idx=conclusion.index)
    graph[conc] = set()
    return lex_anchors, conc


def add_type(graph: Graph, wordtype: WordType, vargen: Iterator[int]) -> Node:

    def fn(wt: WordType, pol: bool) -> Node:
        if isinstance(wt, PolarizedType):
            return ANode(next(vargen), label=wt.depolarize().type, polarity=pol, j_idx=wt.index)
        if isinstance(wt, FunctorType):
            node = CNode(next(vargen), '→')
            argnode = fn(wt.argument, not pol)
            resnode = fn(wt.result, pol)
            pos, neg = (resnode, argnode) if pol else (argnode, resnode)
            graph[node].add(Edge(pos, res))
            graph[neg].add(Edge(node, arg))
            return node
        if isinstance(wt, DiamondType):
            node = CNode(next(vargen), wt.modality)
            argnode = fn(wt.content, pol)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph[src].add(Edge(tgt, arg))
            return node
        if isinstance(wt, BoxType):
            node = CNode(next(vargen), wt.modality)
            argnode = fn(wt.content, pol)
            src, tgt = (node, argnode) if pol else (argnode, node)
            graph[src].add(Edge(tgt, res))
            return node
        if isinstance(wt, EmptyType):
            return ANode(next(vargen), label=wt.type, polarity=pol, j_idx=-1)

    root = fn(wordtype, True)
    return root


def merge_multi_crd(graph: Graph, anchors: List[Node]):
    def daughter(_anchor: Node) -> Node:
        tgts = find_targets(graph, _anchor)
        assert len(tgts) == 1
        return list(tgts)[0]

    empties = []
    for anchor in reversed(anchors):
        root = daughter(anchor)
        if root.label == '_':
            empties.append(root)
        if root.label == 'cnj' and empties:
            empty = empties.pop()
            cnj = daughter(root)
            graph[empty].add(Edge(cnj, lex))


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
        for t, e in graph[s]:
            dg.node(str(s.index) if s is not None else 'SRC', label=str(s))
            dg.node(str(t.index), label=str(t))

            dg.edge(str(s.index) if s is not None else 'SRC', str(t.index), label=e)
    dg.render(view=True)