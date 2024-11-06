from typing import Union
from dataclasses import dataclass

import networkx as nx

from .types import GraphType, GraphType_Literal, LayoutFunction



@dataclass
class Topology_Line:
	length: int

@dataclass
class Topology_Grid:
	dimensions: tuple[int, int]

@dataclass
class Topology_GridDiagonals:
	dimensions: tuple[int, int]

@dataclass
class Topology_Cube:
	dimensions: tuple[int, int, int]

@dataclass
class Topology_Ring:
	length: int

@dataclass
class Topology_Star:
	outer_nodes: int

@dataclass
class Topology_Planar:
	nodes: int

@dataclass
class Topology_Clique:
	nodes: int

@dataclass
class Topology_Torus:
	dimensions: tuple[int, int]

@dataclass
class Topology_Hypertorus:
	dimensions: tuple[int, int, int]

# TODO
# @dataclass
# class Topology_Sphere:
# 	radius: int

# TODO
# @dataclass
# class Topology_Manifold:
# 	pass

@dataclass
class Topology_RandomTree:
	nodes: int

@dataclass
class Topology_RandomErdosRenyi:
	nodes: int
	probability: float

@dataclass
class Topology_RandomBarabasiAlbert:
	nodes: int
	edges: int

@dataclass
class Topology_RandomWattsStrogatz:
	nodes: int
	edges: int
	rewiring_probability: float

TopologyType = Union[
	Topology_Line,
	Topology_Grid,
	Topology_GridDiagonals,
	Topology_Cube,
	Topology_Ring,
	Topology_Star,
	Topology_Planar,
	Topology_Clique,
	Topology_Torus,
	Topology_Hypertorus,
	# Topology_Sphere,
	# Topology_Manifold,
	Topology_RandomTree,
	Topology_RandomErdosRenyi,
	Topology_RandomBarabasiAlbert,
	Topology_RandomWattsStrogatz,
]

"""
TopologyType_Literal = Literal[
	"Line",
	"Grid",
	"Cube",
	"Ring",
	"Star",
	"Planar",
	"Clique",
	"Torus",
	"Hypertorus",
	# "Sphere",
	# "Manifold",
	"RandomTree",
	"RandomErdosRenyi",
	"RandomBarabasiAlbert",
	"RandomWattsStrogatz",
]
"""

@dataclass
class TopologyConfig_Explicit:
	graph : GraphType
	layout: LayoutFunction | None

@dataclass
class TopologyConfig_Generated:
	graph_type    : GraphType_Literal
	topology_type : TopologyType

TopologyConfig = Union[	
	TopologyConfig_Explicit,
	TopologyConfig_Generated,
]

class Topology:
	def __init__(self, config: TopologyConfig):
		match config:
			case TopologyConfig_Explicit(graph, layout):
				self.graph      = graph
				self.get_layout = layout if layout else nx.kamada_kawai_layout
			case TopologyConfig_Generated(graph_type, topology_type):
				self.graph      = self.generate_graph(graph_type, topology_type)
				self.get_layout = Topology.get_layout_function(topology_type)
			case _:
				raise ValueError(f"Unknown topology configuration '{config}'.")

	def generate_graph(
		self,
		graph_type    : GraphType_Literal,
		topology_type : TopologyType,
	) -> GraphType:
		if   graph_type == "Graph"        : graph_builder = nx.Graph
		elif graph_type == "DiGraph"      : graph_builder = nx.DiGraph
		elif graph_type == "MultiGraph"   : graph_builder = nx.MultiGraph
		elif graph_type == "MultiDiGraph" : graph_builder = nx.MultiDiGraph
		else:
			raise ValueError(f"Invalid graph type '{graph_type}'")


		def build_diagonal_grid(dimensions: tuple[int, int]) -> GraphType:
			result = nx.grid_graph(dim=dimensions)
			for i in range(dimensions[0]-1):
				for j in range(dimensions[1]-1):
					result.add_edge((i,j), (i+1,j+1))
					result.add_edge((i+1,j), (i,j+1))
			return result

		match topology_type:
			case Topology_Line                 (length)                             :  topology = nx.path_graph(length)
			case Topology_Grid                 (dimensions)                         :  topology = nx.grid_graph(dim=dimensions)
			case Topology_GridDiagonals        (dimensions)                         :  topology = build_diagonal_grid(dimensions)
			case Topology_Cube                 (dimensions)                         :  topology = nx.grid_graph(dim=dimensions)
			case Topology_Ring                 (length)                             :  topology = nx.cycle_graph(length)
			case Topology_Star                 (outer_nodes)                        :  topology = nx.star_graph(outer_nodes)
			case Topology_Planar               (nodes)                              :  raise NotImplementedError("Topology.generate_graph(): Topology_Planar")
			case Topology_Clique               (nodes)                              :  topology = nx.complete_graph(nodes)
			case Topology_Torus                (dimensions)                         :  topology = nx.grid_graph(dim=dimensions, periodic=True)
			case Topology_Hypertorus           (dimensions)                         :  topology = nx.grid_graph(dim=dimensions, periodic=True)
			case Topology_RandomTree           (nodes)                              :  topology = nx.random_tree(nodes)
			case Topology_RandomErdosRenyi     (nodes, probability)                 :  topology = nx.erdos_renyi_graph(nodes, probability)
			case Topology_RandomBarabasiAlbert (nodes, edges)                       :  topology = nx.barabasi_albert_graph(nodes, edges)
			case Topology_RandomWattsStrogatz  (nodes, edges, rewiring_probability) :  topology = nx.watts_strogatz_graph(nodes, edges, rewiring_probability)
			case _:
				raise ValueError(f"Unknown topology type {topology_type}.")
		topology = nx.relabel_nodes(topology, {node: i for i, node in enumerate(topology.nodes())})
		result = graph_builder(topology)
		return result

	@staticmethod
	def get_layout_function(topology_type: TopologyType) -> LayoutFunction:
		# https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout
		match topology_type:
			case Topology_Line                 (_) :  result = nx.spring_layout
			case Topology_Grid                 (_) :  result = nx.kamada_kawai_layout
			case Topology_GridDiagonals        (_) :  result = nx.kamada_kawai_layout
			case Topology_Cube                 (_) :  result = nx.fruchterman_reingold_layout
			case Topology_Ring                 (_) :  result = nx.spring_layout
			case Topology_Star                 (_) :  result = nx.spring_layout
			case Topology_Planar               (_) :  result = nx.spring_layout
			case Topology_Clique               (_) :  result = nx.spring_layout
			case Topology_Torus                (_) :  result = nx.fruchterman_reingold_layout
			case Topology_Hypertorus           (_) :  result = nx.fruchterman_reingold_layout
			case Topology_RandomTree           (_) :  result = nx.spring_layout
			case Topology_RandomErdosRenyi     (_) :  result = nx.spring_layout
			case Topology_RandomBarabasiAlbert (_) :  result = nx.spring_layout
			case Topology_RandomWattsStrogatz  (_) :  result = nx.spring_layout
			case _:
				raise ValueError("get_layout_function(): Unknown topology type.")
		return result
