from pywhy_graphs import PAG

from typing import Dict, FrozenSet, Iterator, Mapping, Any, Tuple, Union

import networkx as nx

Node = Union[int, float, str, Any]

class PAG_custom(PAG):
    """Partial ancestral graph (PAG).

    PAGs are a Markov equivalence class with mixed edges of directed,
    bidirected, undirected and edges with circle endpoints. In terms
    of graph functionality, they essentially extend the definition of
    an ADMG with edges with circular endpoints.

    Parameters
    ----------
    incoming_directed_edges : input directed edges (optional, default: None)
        Data to initialize directed edges. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    incoming_undirected_edges : input undirected edges (optional, default: None)
        Data to initialize undirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    incoming_bidirected_edges : input bidirected edges (optional, default: None)
        Data to initialize bidirected edges. All arguments that are accepted
        by `networkx.Graph` are accepted.
    incoming_circle_edges : input circular endpoint edges (optional, default: None)
        Data to initialize edges with circle endpoints. All arguments that are accepted
        by `networkx.DiGraph` are accepted.
    directed_edge_name : str
        The name for the directed edges. By default 'directed'.
    undirected_edge_name : str
        The name for the undirected edges. By default 'undirected'.
    bidirected_edge_name : str
        The name for the bidirected edges. By default 'bidirected'.
    circle_edge_name : str
        The name for the circle edges. By default 'circle'.
    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    networkx.DiGraph
    networkx.Graph
    pywhy_graphs.ADMG
    pywhy_graphs.networkx.MixedEdgeGraph

    Notes
    -----
    PAGs are Markov equivalence class of causal ADMGs. The implicit assumption in
    these causal graphs are the Structural Causal Model (or SCM) is Semi-Markovian, such
    that latent confounders may be present. This allows PAGs to be learned from
    constraint-based (such as the FCI algorithm) approaches for causal structure learning.

    **Edge Type Subgraphs**

    The data structure underneath the hood is stored in two types of networkx graphs:
    ``networkx.Graph`` and ``networkx.DiGraph``.

    - Directed edges (<-, ->, indicating causal relationship) = `networkx.DiGraph`
        The subgraph of directed edges may be accessed by the
        `~.PAG.sub_directed_graph`. Their edges in networkx format can be
        accessed by :attr:`~.ADMG.directed_edges` and the corresponding
        name of the edge type by :attr:`~.ADMG.directed_edge_name`.
    - Bidirected edges (<->, indicating latent confounder) = `networkx.Graph`
        The subgraph of bidirected edges may be accessed by the
        `~.PAG.sub_bidirected_graph`. Their edges in networkx format can be
        accessed by :attr:`~.ADMG.bidirected_edges` and the corresponding name of the
        edge type by :attr:`~.ADMG.bidirected_edge_name`.
    - Undirected edges (--, indicating selection bias) = `networkx.Graph`
        The subgraph of undirected edges may be accessed by the
        `~.PAG.sub_undirected_graph`. Their edges in networkx format can be
        accessed by `~.PAG.undirected_edges` and the corresponding name of the
        edge type by `~.PAG.undirected_edge_name`.
    - Circle edges (*-o, o-*, indicating uncertainty) = `networkx.DiGraph`
        The subgraph of undirected edges may be accessed by the
        `~.PAG.sub_circle_graph`. Their edges in networkx format can be
        accessed by `~.PAG.circle_edges` and the corresponding name of the
        edge type by `~.PAG.circle_edge_name`.

    **How different edges are represented in the PAG**

    Compared to an `~pywhy_graphs.classes.ADMG` and `~pywhy_graphs.classes.CPDAG` and a
    :class:`networkx.DiGraph`, a PAG is more complex in that it generalizes endpoints an edge can
    take, exponentially increasing the number of possible edges that can occur between two
    nodes. The main complication arises in edges with circle endpoints. Rather than store all
    possible edges as separate networkx graphs, we have a set of rules that map a combination
    of the above edge-type subgraphs to a certain edge.

    Bidirected and undirected edges are represented by one networkx graph (`networkx.Graph`).
    They are simple in that they do not require pairing with another edge-type subgraph.

    - ``x <-> y``: is a bidirected edge present? (Note by definition of a PAG no other edge
        can be present between x and y)
    - ``x - y``: is an undirected present? (Note no other edge should be present in any
        other direction, so an undirected edge is similar to a bidirected edge in that it
        represents only one kind of edge)

    Edges with arrowheads, tails and circular endpoints are represented by another networkx
    graph (`networkx.DiGraph`). They complicate matters because the
    `~.PAG.sub_directed_graph` and
    `~.PAG.sub_circle_graph` can be combined in different ways to
    result in different edges between x and y.

    Without loss of generality, we will be dealing with the ordered tuple (x, y). If you want
    the other direction of the edge, you can just flip the order of x and y. For example,
    ``x <- y`` would just be ``y -> x``, so we will only discuss the ``->`` edge. The following
    rules dictate what sort of edge we are dealing with:

    - ``x o-o y``: is circle edge present in both directions? There are **only** edges present
        in the `~.PAG.sub_circle_graph` between x and y.
    - ``x o-> y``: is circle edge one way and directed edge another way? There is an edge from
        the `~.PAG.sub_circle_graph` and the
        `~.PAG.sub_directed_graph` between x and y in opposite
        directions.
    - ``x o- y``: is there only one circle edge? In this special case, we do not use the
        `~.PAG.sub_undirected_graph` to represent the tail endpoint at y. There is **only**
        one edge in the `~.PAG.sub_circle_graph` between x and y.
    """

    def __init__(
        self,
        incoming_directed_edges=None,
        incoming_undirected_edges=None,
        incoming_bidirected_edges=None,
        incoming_circle_edges=None,
        incoming_associated_edges=None,
        directed_edge_name: str = "directed",
        undirected_edge_name: str = "undirected",
        bidirected_edge_name: str = "bidirected",
        circle_edge_name: str = "circle",
        associated_name="associated",
        **attr,
    ):
        super().__init__(
            incoming_directed_edges=incoming_directed_edges,
            incoming_undirected_edges=incoming_undirected_edges,
            incoming_bidirected_edges=incoming_bidirected_edges,
            incoming_circle_edges=incoming_circle_edges,
            directed_edge_name=directed_edge_name,
            undirected_edge_name=undirected_edge_name,
            bidirected_edge_name=bidirected_edge_name,
            circle_edge_name=circle_edge_name,
            **attr,
        )
        # add associated edges
        self.add_edge_type(nx.Graph(incoming_associated_edges), associated_name)
        self._associated_name = associated_name
        
        # extended patterns store unfaithful triples
        # these can be used for conservative structure learning algorithm
        self._unfaithful_triples: Dict[FrozenSet[Node], None] = dict()

        # check that construction of PAG was valid
        from pywhy_graphs import is_valid_mec_graph

        is_valid_mec_graph(self)

    @property
    def associated_edge_name(self) -> str:
        """Name of the associated edge subgraph."""
        return self._associated_name

    @property
    def associated_edges(self) -> Mapping:
        """``EdgeView`` of the associated edges."""
        return self.get_graphs(self.associated_edge_name).edges
    
    def sub_associated_graph(self) -> nx.Graph:
        """Sub-graph of just the associated edges."""
        return self._get_internal_graph(self.associated_edge_name)

    def orient_uncertain_edge(self, u: Node, v: Node) -> None:
        """Orient undirected edge into an arrowhead.

        If there is an undirected edge u - v, then the arrowhead
        will orient u -> v. If the correct order is v <- u, then
        simply pass the arguments in different order.

        Parameters
        ----------
        u : node
            The parent node
        v : node
            The node that 'u' points to in the graph.
        """
        if not self.has_edge(u, v, self.circle_edge_name):
            raise RuntimeError(f"There is no uncertain circular edge between {u} and {v}.")

        # Performs orientation of edges
        if self.has_edge(v, u, self.directed_edge_name):
            # Orients: u <-o v => u <-> v
            # when we orient (u,v) now as an arrowhead, it is a bidirected arrow
            self.remove_edge(v, u, self.directed_edge_name)
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.bidirected_edge_name)
        elif self.has_edge(v, u, self.circle_edge_name):
            # Orients: u o-o v => u o-> v
            # In this case, we have a bidirected circle edge
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.directed_edge_name)
        elif self.has_edge(u, v, self.circle_edge_name):
            # In this case, we have a circle edge that is oriented into an arrowhead
            # we only need to remove the circle edge and orient
            # it as a normal edge
            self.remove_edge(u, v, self.circle_edge_name)
            self.add_edge(u, v, self.directed_edge_name)
        # elif self.has_edge(u, v, self.associated_edge_name):
        #     self.remove_edge(u, v, self.associated_edge_name)
        #     self.add_edge(u, v, self.directed_edge_name)
        else:  # noqa
            raise RuntimeError("The current PAG is invalid.")
