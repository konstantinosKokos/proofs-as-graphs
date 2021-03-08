from ..typing import (Tuple, List, Iterator, Set, Dict, Callable, Node, ANode, CNode, WNode, Graph, Edge, GraphData,
                      Maybe, Iterable)
from collections import defaultdict
from itertools import count


from LassyExtraction.milltypes import (WordType, FunctorType, DiamondType, BoxType, EmptyType,
                                       AtomicType, print_box, print_diamond)
from LassyExtraction.aethel import ProofNet
from LassyExtraction.terms import (Term, Var, Lex, Application, Abstraction, DiamondElim, DiamondIntro, BoxElim,
                                   BoxIntro, cap, wedge, cup, vee)


def tokenize_data(untokenized: GraphData, atom_map: Callable[[List[str]], List[int]],
                  word_map: Callable[[List[Tuple[str, bool]]], List[Tuple[int, bool]]]) -> Maybe[GraphData]:
    edge_map = {'l': 0, 'r': 1, 'u': 2}
    atom_ids = atom_map(untokenized.nodes)
    ids_and_starts = word_map(untokenized.words)
    l1, l2 = len([i for i, s in ids_and_starts if s]), len(untokenized.roots)
    if l1 != l2:
        return None
    return GraphData(words=ids_and_starts, nodes=atom_ids, edge_index=untokenized.edge_index,
                     roots=untokenized.roots, edge_attrs=[edge_map[edge] for edge in untokenized.edge_attrs])


def proofnet_to_graphdata(proofnet: ProofNet) -> Maybe[GraphData]:
    _words = [w.strip('.,').replace('_', ' ') for w in proofnet.proof_frame.get_words()]
    _types = proofnet.proof_frame.get_types()
    graph = defaultdict(lambda: set())
    add_term(graph, proofnet.get_term(), count())
    lex_anchors = get_anchors(graph)    # WNodes, listed in order of term traversal
    if len([w for w, t in zip(_words, _types) if not isinstance(t, EmptyType)]) != len(lex_anchors):
        return None

    # Now convert to graphdata
    nodes = sort_nodes(get_nodes(graph))
    node_dict = {node.index: i for i, node in enumerate(nodes)}
    srcs, tgts, attrs = [], [], []
    for k in nodes:
        vs = sort_edges(graph[k])
        srcs.extend([node_dict[k.index] for _ in graph[k]])
        tgts.extend([node_dict[v.index] for v, _ in vs])
        attrs.extend([e for _, e in vs])
    roots = [i for i, n in enumerate(nodes) if isinstance(n, WNode)]
    return GraphData(nodes=['[PAD]' if isinstance(node, WNode) else node.label for node in nodes],
                     edge_index=(srcs, tgts), words=[(w, isinstance(t, EmptyType)) for w, t in zip(_words, _types)],
                     edge_attrs=attrs, roots=roots)


def sort_edges(edges: Iterable[Edge]) -> List[Edge]:
    wedges = sorted([e for e in edges if isinstance(e.target, WNode)], key=lambda e: int(e.target.label))
    other = sorted([e for e in edges if not isinstance(e.target, WNode)], key=lambda e: e.target.index)
    return wedges + other


def sort_nodes(nodes: Iterable[Node]) -> List[Node]:
    wnodes = sorted([n for n in nodes if isinstance(n, WNode)], key=lambda n: int(n.label))
    other = sorted([n for n in nodes if not isinstance(n, WNode)], key=lambda n: n.index)
    return wnodes + other


def merge_multi_crd(words: List[str], types: List[WordType]) -> Tuple[List[str], List[WordType]]:
    # distinguish between crd and det case
    rw, rt = [], []
    empties = []
    adj = False
    for w, t in zip(reversed(words), reversed(types)):
        if isinstance(t, EmptyType):
            empties.append(w)
            adj = True
        elif empties:
            if adj and isinstance(t, BoxType) and t.modality == 'det':
                e = empties.pop()
                rw.append(f'{w} {e}')
                rt.append(t)
                adj = False
            elif isinstance(t, FunctorType) and isinstance(t.argument, DiamondType) and t.argument.modality == 'cnj':
                e = empties.pop()
                rw.append(f'{w} {e}')
                rt.append(t)
            else:
                rw.append(w)
                rt.append(t)
                adj = False
        else:
            rw.append(w)
            rt.append(t)
            adj = False
    return list(reversed(rw)), list(reversed(rt))


def make_atom_map(graphs: List[GraphData]) -> Dict[str, int]:
    labels = set.union(*[set(g.nodes) for g in graphs]).difference({'[PAD]'})
    return {label: i for i, label in enumerate(['[PAD]'] + sorted(labels))}


def get_nodes(graph: Graph) -> Set[Node]:
    return set.union(*[set([k] + [n for n, _ in graph[k]]) for k in graph.keys()])


def get_anchors(graph: Graph) -> List[Node]:
    return sorted(filter(lambda n: isinstance(n, WNode), get_nodes(graph)), key=lambda n: int(n.label))


def ledge(x: Node) -> Edge:
    return Edge(x, 'l')


def redge(x: Node) -> Edge:
    return Edge(x, 'r')


def uedge(x: Node) -> Edge:
    return Edge(x, 'u')


def add_term(graph: Graph, term: Term, vargen: Iterator[int]) -> Node:
    if isinstance(term, Application):
        app_node = CNode(next(vargen), 'A')
        graph[app_node].add(ledge(add_term(graph, term.functor, vargen)))
        graph[app_node].add(redge(add_term(graph, term.argument, vargen)))
        return app_node
    elif isinstance(term, Abstraction):
        abs_node = CNode(next(vargen), 'L')
        graph[abs_node].add(ledge(add_term(graph, term.abstraction, vargen)))
        graph[abs_node].add(redge(add_term(graph, term.body, vargen)))
        return abs_node
    elif isinstance(term, Var):
        var_node = CNode(next(vargen), 'V')
        graph[var_node].add(ledge(add_type(graph, term.type(), vargen)))
        graph[var_node].add(redge(ANode(next(vargen), str(term.idx))))
        return var_node
    elif isinstance(term, Lex):
        con_node = CNode(next(vargen), 'C')
        graph[con_node].add(ledge(add_type(graph, term.type(), vargen)))
        graph[con_node].add(redge(WNode(next(vargen), str(term.idx))))
        return con_node
    elif isinstance(term, DiamondElim):
        return add_term(graph, term.body, vargen)
        # mod_node = CNode(next(vargen), vee(term.diamond))
        # graph[mod_node].add(uedge(add_term(graph, term.body, vargen)))
        # return mod_node
    elif isinstance(term, DiamondIntro):
        return add_term(graph, term.body, vargen)
        # mod_node = CNode(next(vargen), wedge(term.diamond))
        # graph[mod_node].add(uedge(add_term(graph, term.body, vargen)))
        # return mod_node
    elif isinstance(term, BoxElim):
        return add_term(graph, term.body, vargen)
        # mod_node = CNode(next(vargen), cup(term.box))
        # graph[mod_node].add(uedge(add_term(graph, term.body, vargen)))
        # return mod_node
    elif isinstance(term, BoxIntro):
        return add_term(graph, term.body, vargen)
        # mod_node = CNode(next(vargen), cap(term.box))
        # graph[mod_node].add(uedge(add_term(graph, term.body, vargen)))
        # return mod_node
    else:
        raise TypeError


def add_type(graph: Graph, wordtype: WordType, vargen: Iterator[int]) -> Node:
    return ANode(next(vargen), label=str(wordtype.depolarize()))
    # if isinstance(wordtype, AtomicType):
    #     return ANode(next(vargen), label=wordtype.depolarize().type)
    # if isinstance(wordtype, FunctorType):
    #     fun_node = CNode(next(vargen), 'fun')
    #     graph[fun_node].add(ledge(add_type(graph, wordtype.argument, vargen)))
    #     graph[fun_node].add(redge(add_type(graph, wordtype.result, vargen)))
    #     return fun_node
    # if isinstance(wordtype, BoxType):
    #     m_node = CNode(next(vargen), print_box(wordtype.modality))
    #     graph[m_node].add(uedge(add_type(graph, wordtype.content, vargen)))
    #     return m_node
    # if isinstance(wordtype, DiamondType):
    #     m_node = CNode(next(vargen), print_diamond(wordtype.modality))
    #     graph[m_node].add(uedge(add_type(graph, wordtype.content, vargen)))
    #     return m_node
    # if isinstance(wordtype, EmptyType):
    #     raise NotImplementedError
    # raise TypeError(f'Wtf is this {wordtype}')


def visualize(graph: Graph):
    import graphviz as gv

    dg = gv.Digraph()

    for s in graph.keys():
        for t, e in graph[s]:
            if t is None:
                print(s)
                continue
            dg.node(str(s.index), label=str(s))
            dg.node(str(t.index), label=str(t))
            dg.edge(str(s.index), str(t.index), label=e)
    dg.render(view=True)