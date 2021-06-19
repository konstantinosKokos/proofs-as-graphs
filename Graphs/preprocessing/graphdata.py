from ..types import (GraphData, Graph, List, ANode, WNode, Node, Tuple, Iterator, TNode, CNode, Maybe, Union,
                     Set, Edge, Dict, Callable)
from LassyExtraction.aethel import (ProofNet, AxiomLinks)

from LassyExtraction.milltypes import (WordType, PolarizedType, FunctorType, DiamondType, BoxType, EmptyType)

from collections import defaultdict
from itertools import count


def proofnet_to_graphdata(pn: ProofNet) -> Maybe[GraphData]:
    # Create graph structure
    words = pn.proof_frame.get_words()
    types = pn.proof_frame.get_types()
    graph: Graph = defaultdict(lambda: set())
    lex_anchors, conclusion = add_types(graph, words, types, pn.proof_frame.conclusion)
    conc = add_axiom_links(graph, conclusion, pn.axiom_links)
    try:
        merge_multi_crd(graph, lex_anchors)
    except ValueError:
        return None

    # Now convert to graphdata
    nodes = sorted(get_nodes(graph), key=lambda node: node.index)
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    srcs, tgts, attrs = [], [], []
    for k in nodes:
        vs = sorted(graph[k], key=lambda edge: edge.target.index)
        srcs.extend([node_dict[k.index] for _ in graph[k]])
        tgts.extend([node_dict[v.index] for v, _ in vs])
        attrs.extend([e for _, e in vs])

    return GraphData(nodes=['[LEX]' if isinstance(node, WNode) else node.label for node in nodes],
                     edge_index=(srcs, tgts), words=words, edge_attrs=attrs, conclusion=node_dict[conc.index],
                     roots=sorted([node_dict[lanc.index] for lanc in lex_anchors]))


def add_types(graph: Graph, words: List[str], wordtypes: List[WordType], conclusion: PolarizedType) \
        -> Tuple[List[WNode], ANode]:

    def make_anchor(_w: str) -> WNode:
        return WNode(next(counter), _w)

    counter = count()
    wordtypes = list(map(collate_type, wordtypes))
    lex_anchors = []
    for word, wordtype in zip(words, wordtypes):
        anchor = make_anchor(word)
        lex_anchors.append(anchor)
        root = add_type(graph, wordtype, counter)
        graph[anchor].add(Edge(root, 'lex'))
    conc = ANode(next(counter), label=conclusion.depolarize().type, polarity=False, link_idx=conclusion.index)
    conc_anchor = make_anchor('[CLS]')
    graph[conc_anchor].add(Edge(conc, 'sum'))
    graph[conc] = set()
    return [conc_anchor] + lex_anchors, conc


def add_type(graph: Graph, wordtype: WordType, vargen: Iterator[int]) -> Node:

    def add(_g: Graph, src: Node, tgt: Node, label: Maybe[str]):
        if label is None:
            label = '-'
        if src != tgt:
            _g[src].add(Edge(tgt, label))

    def fn(wt: WordType, pol: bool, connective: Maybe[Union[CNode, TNode]] = None, ctx: Maybe[str] = None) \
            -> Tuple[Node, Maybe[str]]:
        if isinstance(wt, PolarizedType):
            return ANode(next(vargen), label=wt.depolarize().type, polarity=pol, link_idx=wt.index), None
        if isinstance(wt, FunctorType):
            connective = (TNode(next(vargen)) if pol else CNode(next(vargen))) if connective is None else connective
            argnode, arglabel = fn(wt.argument, not pol)
            resnode, reslabel = fn(wt.result, pol, connective)
            arglabel = arglabel or ctx
            arg, res = (argnode, arglabel), (resnode, reslabel)
            (pnode, plabel), (nnode, nlabel) = (res, arg) if pol else (arg, res)
            add(graph, connective,  pnode, plabel)
            add(graph, nnode, connective, nlabel)
            return connective, None
        if isinstance(wt, DiamondType):
            argnode, _ = fn(wt.content, pol)
            return argnode, wt.modality
        if isinstance(wt, BoxType):
            return fn(wt.content, pol, ctx=wt.modality)
        if isinstance(wt, EmptyType):
            return ANode(next(vargen), label=wt.type, polarity=pol, link_idx=-1), None

    root, _ = fn(wordtype, True)
    return root


def collate_type(t: WordType) -> WordType:
    return t


def visualize(graph: Graph):
    import graphviz as gv

    dg = gv.Digraph()

    for s in list(graph.keys()):
        for t, l in list(graph[s]):
            dg.node(str(s.index) if s is not None else 'SRC', label=str(s))
            dg.node(str(t.index), label=str(t))
            dg.edge(str(s.index) if s is not None else 'SRC', str(t.index), label=l)
    dg.render(view=True)


def add_axiom_links(graph: Graph, cnode: ANode, axiom_links: AxiomLinks) -> ANode:
    for _pos, _neg in axiom_links:
        pos = find_by_link_idx(graph, _pos)
        neg = find_by_link_idx(graph, _neg)
        if neg == cnode:
            cnode = pos
        sources = find_sources(graph, neg)
        targets = graph[neg]
        graph[pos] = graph[pos].union(targets)
        if sources:
            assert len(sources) == 1
            anchor, edge = list(sources)[0]
            assert edge.label == 'sum'
            graph[anchor] = {Edge(pos, 'sum')}
        del graph[neg]
    return cnode


def get_nodes(graph: Graph) -> Set[Node]:
    return set(graph.keys()).union(*[find_targets(graph, k) for k in graph.keys()])


def find_sources(graph: Graph, target: Node) -> Set[Tuple[Node, Edge]]:
    return {(source, edge) for source in graph.keys() for edge in graph[source] if edge.target == target}


def find_targets(graph: Graph, source: Node) -> Set[Node]:
    return {n for n, _ in graph[source]}


def find_by_link_idx(graph: Graph, link_idx: int) -> ANode:
    candidates = [node for node in get_nodes(graph) if isinstance(node, ANode) and node.link_idx == link_idx]
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} nodes with {link_idx=}: {candidates}')
    return candidates[0]


def daughter(graph: Graph, anchor: Node) -> Node:
    tgts = find_targets(graph, anchor)
    assert len(tgts) == 1
    return list(tgts)[0]


def merge_multi_crd(graph: Graph, anchors: List[WNode]):
    empties, adj = [], False
    for anchor in reversed(anchors):
        root = daughter(graph, anchor)
        if root.label == '_':
            empties.append(anchor)
            adj = True
        elif empties:
            if any([edge.label == 'cnj' for edge in graph[anchor]]):
                empty = empties.pop()
                graph[empty].add(Edge(root, 'lex'))
            elif adj and any([edge.label == 'det' for src, edge in find_sources(graph, root)]):
                empty = empties.pop()
                graph[empty] = {Edge(root, 'lex')}
            adj = False
    if empties:
        raise ValueError


def get_vocab(graphs: List[GraphData]) -> Tuple[Dict[str, int], Dict[str, int]]:
    labels = set.union(*[set(g.nodes) for g in graphs]).difference({'[LEX]', '[MASK]'})
    attrs = set.union(*[set(g.edge_attrs) for g in graphs])
    return {label: i for i, label in enumerate(['[LEX]', '[MASK]'] + sorted(labels))}, \
           {attr: i for i, attr in enumerate(attrs)}


def tokenize_graph(graph: GraphData, node_map: Dict[str, int], edge_map: Dict[str, int],
                   word_map: Callable[[List[str]], List[Tuple[int, bool]]]) -> GraphData:
    node_ids = [node_map[node] for node in graph.nodes]
    edge_ids = [edge_map[attr] for attr in graph.edge_attrs]
    return GraphData(words=word_map(graph.words), nodes=node_ids, edge_index=graph.edge_index,
                     edge_attrs=edge_ids, conclusion=graph.conclusion, roots=graph.roots)


# def remove_anchors(graph: Graph, anchors: List[Node]) -> List[Node]:
#     ret = []
#     for anchor in anchors:
#         ret.append(daughter(graph, anchor))
#         del graph[anchor]
#     return ret
#
#
# def collate_atom(atom: str) -> str:
#     return 'NP' if atom == 'SPEC' else atom
#
#
# def collate_type(wordtype: WordType) -> WordType:
#     if isinstance(wordtype, AtomicType):
#         wordtype.type = collate_atom(wordtype.type)
#         return wordtype
#     elif isinstance(wordtype, FunctorType):
#         collate_type(wordtype.argument)
#         collate_type(wordtype.result)
#         return wordtype
#     elif isinstance(wordtype, ModalType):
#         collate_type(wordtype.content)
#         return wordtype
#     raise TypeError(f'Unexpected argument {wordtype} of type {type(wordtype)}')
