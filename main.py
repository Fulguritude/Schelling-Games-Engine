"""
Schelling games engine

Initial goal design:

The following is a list of kinds of Schelling games that should be possible to render

Model features
- accepts arbitrary (di)graph G(V,E) topologies
- n agents with n ≤ |V|
- accepts only first degree neighborhood
- agent natures are static
- handles T a set of t discrete and continuous agent types Tⱼ
	-> discrete categories (race, religion)
	-> mixed discrete/continuous: discrete partition of unity (eg mixed-raceness; multicultural identity; eg 25% black, 75% white); fundamentally this is just multi-continuous but with a constraint
	-> continuous (income, extraversion, etc)
- each agent i described by a vector of its types (vector Vᵢ of values in types Tⱼ)
- vector of agent utility / type preferences (vector τ of τⱼ) on the properties of their neighborhood
	-> for discrete preference: minimal (desire for similarity/homophily) and maximal τ (desire for integration/heterophily)
	-> for mixed preference: minimal/maximal degree of the aspects of the makeup
	-> for continuous preference: range around current "income"
	=> arbitrary lambdas (?)
- random, strategic and stubborn agent natures
- can add node and edge weights and labels
- digraphs to model asymmetric relationships
- social graphs (based on a secondary digraph with both friend and enemies)
- max iteration and equilibrium halts
- configurable agent makeup (type proportions, distributions, etc)
- move decision are taken over a fixed state (agents must wait for some node to be freed by all agents having acted, before considering them)

UI features
- type-based color toggles
- iteration slider (store history option)
- palette choices
- special representations for grids or regular-ish graphs ?

Graph generators:
- stars
- cliques
- trees
- nD grids (lines, squares, cubes, etc)
- simple manifold graphs (ring, sphere, torus, etc)
- planar graphs, "manifold-planar" graphs ? (would be cool to have an Earth density-based graph generator for international migration)

Considered (maybe for future versions):
- Schelling games on ultragraphs (for regional cultural norms)
- dynamic evolution (beliefs, interests, opinions, due to random neighborhood influence)
- Ways to integrate physical clustering models (Vinkovic/Kirman 2006)
- Integrating richer neighborhood rules (like cellular automata; a partially non-local automaton model)
- metrics for the graph: PoA, PoS, segregation/integration degree,
- multiple agents per node (Chan et alii 2020)

Tools:
- Model: networkx
- UI frontend: Kivy
- Considered for performance gains: GPU backend: pygfx and VBO for backend
- Considered for improved graph rendering: https://pygraphviz.github.io/documentation/stable/install.html

Installation:
- `pip3 install networkx kivy matplotlib kivy_garden.matplotlib`


"""

from __future__ import annotations
from typing import Any, Union, Literal, Hashable, Callable, TypedDict, Iterable, get_args, cast
from os import mkdir
from os.path import exists, sep
from dataclasses import dataclass
import random
from math import log
from time import sleep

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors      import Colormap
from kivy.app               import App
from kivy.uix.boxlayout     import BoxLayout
from kivy.uix.slider        import Slider
from kivy.uix.button		import Button
from kivy.uix.togglebutton  import ToggleButton
from kivy.uix.label         import Label
from kivy_garden.matplotlib import FigureCanvasKivyAgg
from PIL import Image




"""
#######################
General types
#######################
"""

Color = str # hex or name
Position = tuple[float, float]


"""
#######################
Agent-relative types
#######################
"""


AgentID                    = int     # unique identifier for an agent
AgentType_Name             = str     # name of a given type, e.g., "race", "religion", "income"
AgentType_Value_Discrete   = int|str # a single value for an ordinal or categorical type
AgentType_Value_Continuous = float   # a single value for a continuous type
AgentType_Value            = Union[  # a single value
	AgentType_Value_Discrete,
	AgentType_Value_Continuous
] 
AgentType_Categories	   = list[AgentType_Value_Discrete]  # a list of possible values for an ordinal or categorical type; set.__get_item__ doesn't exist, so we have a list instead
AgentType_Range			   = tuple[ # a range defining an interval
	AgentType_Value_Continuous,
	AgentType_Value_Continuous,
]
AgentType_Vector           = dict[ # a vector of values for each type, indexed by type name
	AgentType_Name,
	AgentType_Value,
]
AgentType_Domain           = dict[ # a domain of possible values for each type, indexed by type name
	AgentType_Name,
	AgentType_Categories | AgentType_Range
]
AgentType_ValueConstraint  = Union[ # a function that tests a single value
	Callable[[AgentType_Value_Discrete   ], bool],
	Callable[[AgentType_Value_Continuous ], bool],
]
AgentType_VectorConstraint  = Callable[[AgentType_Vector], bool] # a function that tests a vector of values
AgentType_Constraints       = Callable[[AgentType_Domain, AgentType_Vector], bool] # a function that tests a vector of values

AgentType_ColorMap = dict[
	AgentType_Name,
	dict[AgentType_Value_Discrete, Color] | Colormap
]

AgentType_Nature = Literal["random", "strategic", "stubborn"]
AgentType_NatureProportions = TypedDict(
	"AgentType_NatureProportions",
	{
		"random"   : float,
		"strategic": float,
		"stubborn" : float,
	}
)


"""
#######################
Game types
#######################
"""


MovementMode = Literal[
	"jump",      # selects free nodes randomly, moves to the first improvement to utility found
	"swap",      # selects occupied nodes randomly, checks if swap is mutually beneficial, swaps with the first such
#	"mixed",     # runs both jump and swap, TODO
	"max_jump",  # checks all free nodes, moves to the best improvement
	"max_swap",  # naive, checks all occupied nodes for bilateral utility improvement, swaps with the agent with the selfishly best mutually beneficial improvement (only one side is maximized, both sides are improvements)
#	"max_mixed", # runs both max_jump and max_swap, picks the best TODO
#	"mutual_max_swap": #costly, TODO? handle cycles of length > 2 ?
]


"""
#######################
Utility types
#######################
"""

Utility_Criterion_AgentID  = Callable[
	[
		AgentID,
		list[AgentID],
		Any,
	],
	float,
]
Utility_Criterion_Discrete  = Callable[
	[
		AgentType_Value_Discrete,
		list[AgentType_Value_Discrete],
		Any,
	],
	float,
]
Utility_Criterion_Continuous = Callable[
	[
		AgentType_Value_Continuous,
		list[AgentType_Value_Continuous],
		Any,
	],
	float,
]
Utility_Criterion = Union[
	Utility_Criterion_AgentID,
	Utility_Criterion_Discrete,
	Utility_Criterion_Continuous,
]
Utility_Scalarized = Callable[  # cost-type utility for minimization
	[
		AgentType_Vector,       # current node's defining info
		list[AgentType_Vector], # neighbors' defining info
		Any,                    # used for additional context, like for social game graphs
	],
	float,
]



"""
#######################
Graph types
#######################
"""

NodeID    = int   # unique identifier for a node in a given graph
GraphType = Union[ # any valid networkx graph type
	nx.Graph,
	nx.DiGraph,
	nx.MultiGraph,
	nx.MultiDiGraph,
]
NodePosDict    = dict[Hashable, Position]
# example: `pos : NodePosDict = {node: (x, y) for node, (x, y) in zip(graph.nodes(), custom_positions)}`
LayoutFunction = Callable[[GraphType], NodePosDict]

Assignment = dict[AgentID, NodeID]
History    = list[Assignment]

GraphType_Literal = Literal[
	"Graph",
	"DiGraph",
	"MultiGraph",
	"MultiDiGraph",
]


"""
#######################
Color utils
#######################
"""

DEFAULT_COLORS_LIST = [
	"red",
	"green",
	"blue",
	"yellow",
	"orange",
	"purple",
	"cyan",
	"magenta",
	"lime",
	"pink",
	"teal",
	"lavender",
	"brown",
	"beige",
	"maroon",
	"mint",
	"olive",
	"apricot",
	"navy",
	"grey",
	"white",
	"black",
]

DefaultColormapLiterals = Literal[
	"magma",
	"magma_r",
	"viridis",
	"viridis_r",
	"inferno",
	"inferno_r",
	"plasma",
	"plasma_r",
	"cividis",
	"cividis_r",
	"twilight",
	"twilight_r",
	"twilight_shifted",
	"twilight_shifted_r",
	"turbo",
	"turbo_r",
	"mako",
	"mako_r",
	"rocket",
	"rocket_r",
	"icefire",
	"icefire_r",
]
DEFAULT_COLORMAPS : tuple[DefaultColormapLiterals, ...] = get_args(DefaultColormapLiterals)
# usage: plt.get_cmap("magma")  #type:ignore


def get_default_colormap(
	domain   : AgentType_Domain,
	colormap : AgentType_ColorMap | None = None,
) -> AgentType_ColorMap:
	result : AgentType_ColorMap = {}
	for name, values in domain.items():
		if colormap and name in colormap:
			result[name] = colormap[name]
		elif isinstance(values, list):
			result[name] = {
				value: DEFAULT_COLORS_LIST[i]
				for i, value in enumerate(values)
			}
		else:
			result[name] = plt.get_cmap("plasma")  #type:ignore
	return result



"""
#######################
Network topology types
and utils
#######################
"""

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



"""
#######################
Probability distribution
types and utils
#######################
"""

@dataclass
class Distribution_CustomDiscrete:
	proportions : dict[str, float]

@dataclass
class Distribution_UniformDiscrete:
	min_val : float
	max_val : float

@dataclass
class Distribution_UniformContinuous:
	min_val : float
	max_val : float

@dataclass
class Distribution_Normal:
	mean   : float
	stddev : float

@dataclass
class Distribution_LogNormal:
	mean   : float
	stddev : float

@dataclass
class Distribution_Exponential:
	lambda_ : float

@dataclass
class Distribution_Beta:
	alpha : float
	beta  : float

@dataclass
class Distribution_Gamma:
	alpha : float
	beta  : float

@dataclass
class Distribution_Weibull:
	alpha : float
	beta  : float

@dataclass
class Distribution_Triangular:
	low  : float
	high : float
	mode : float

@dataclass
class Distribution_Choice_Values:
	values : list[float]

@dataclass
class Distribution_Choice_Categories:
	categories : list[str]

DistributionType = Union[
	Distribution_CustomDiscrete,
	Distribution_UniformDiscrete,
	Distribution_UniformContinuous,
	Distribution_Normal,
	Distribution_LogNormal,
	Distribution_Exponential,
	Distribution_Beta,
	Distribution_Gamma,
	Distribution_Weibull,
	Distribution_Triangular,
	Distribution_Choice_Values,
	Distribution_Choice_Categories,
]
DistributionGenerator = Union[
	Callable[[], AgentType_Value_Discrete],
	Callable[[], AgentType_Value_Continuous],
]
DomainDistributions = dict[AgentType_Name, DistributionGenerator]

def distribution_type_to_generator(distribution: DistributionType) -> DistributionGenerator:
	result : DistributionGenerator
	match distribution:
		case Distribution_CustomDiscrete    (proportions)      :  result = lambda : random.choices(list(proportions.keys()), weights=list(proportions.values()), k=1)[0]
		case Distribution_UniformDiscrete   (min_val, max_val) :  result = lambda : random.uniform(min_val, max_val)
		case Distribution_UniformContinuous (min_val, max_val) :  result = lambda : random.uniform(min_val, max_val)
		case Distribution_Normal            (mean, stddev)     :  result = lambda : random.normalvariate(mean, stddev)
		case Distribution_LogNormal         (mean, stddev)     :  result = lambda : random.lognormvariate(mean, stddev)
		case Distribution_Exponential       (lambda_)          :  result = lambda : random.expovariate(lambda_)
		case Distribution_Beta              (alpha, beta)      :  result = lambda : random.betavariate(alpha, beta)
		case Distribution_Gamma             (alpha, beta)      :  result = lambda : random.gammavariate(alpha, beta)
		case Distribution_Weibull           (alpha, beta)      :  result = lambda : random.weibullvariate(alpha, beta)
		case Distribution_Triangular        (low, high, mode)  :  result = lambda : random.triangular(low, high, mode)
		case Distribution_Choice_Values     (values)           :  result = lambda : random.choice(values)
		case Distribution_Choice_Categories (categories)       :  result = lambda : random.choice(categories)
		case _:
			raise ValueError(f"Unknown distribution type '{distribution}'.")
	return result

AgentType_Distributions = dict[AgentType_Name, DistributionType]




"""
#######################
Utility type and utils
(probably the worst
part of the code)
#######################
"""

#For all the below, if None, does not constrain
# Absolute refers to the count of valid neighbors
# Relative refers to the ratio of valid neighbors over all neighbors
@dataclass
class Utility_SameNeighbor_Absolute:
	count_min : int | None
	count_max : int | None
	weight    : float | None

@dataclass
class Utility_SameNeighbor_Relative:
	tau_min : float | None
	tau_max : float | None
	weight  : float | None

@dataclass
class Utility_NeighborInRange_Additive_Absolute:
	max_dist  : float
	count_min : int | None
	count_max : int | None
	weight    : float | None

@dataclass
class Utility_NeighborInRange_Additive_Relative:
	max_dist : float
	tau_min  : float | None
	tau_max  : float | None
	weight   : float | None

@dataclass
class Utility_NeighborInRange_Multiplicative_Absolute:
	log_max_dist  : float
	log_count_min : int | None
	log_count_max : int | None
	weight        : float | None

@dataclass
class Utility_NeighborInRange_Multiplicative_Relative:
	log_max_dist : float
	log_tau_min  : float | None
	log_tau_max  : float | None
	weight       : float | None

@dataclass
class Utility_FriendAndEnemies_Absolute:
	handle_friends : bool  # weight_friends : float | None ??
	handle_enemies : bool  # weight_enemies : float | None ??

@dataclass
class Utility_FriendAndEnemies_Relative:
	handle_friends : bool  # weight_friends : float | None ??
	handle_enemies : bool  # weight_enemies : float | None ??

@dataclass
class Utility_SpecificPhily:
	philys : dict[AgentType_Value_Discrete, Utility_Criterion_Discrete]

UtilityType = Union[
	Utility_SameNeighbor_Absolute,
	Utility_SameNeighbor_Relative,
	Utility_NeighborInRange_Additive_Absolute,
	Utility_NeighborInRange_Additive_Relative,
	Utility_NeighborInRange_Multiplicative_Absolute,
	Utility_NeighborInRange_Multiplicative_Relative,
	Utility_FriendAndEnemies_Absolute,
	Utility_FriendAndEnemies_Relative,
	Utility_SpecificPhily,
]



def get_default_utility_criterion_function(utility_type : UtilityType) -> Utility_Criterion:

	def builder_utility_sameneighbor_absolute(
		count_min : int | None,
		count_max : int | None,
		weight    : float | None,
	) -> Utility_Criterion_Discrete:
		def utility_sameneighbor_absolute(
			self_value    : AgentType_Value_Discrete,
			neighbor_vals : list[AgentType_Value_Discrete],
			context       : None,
		) -> float:
			count = sum(1 for neighbor_val in neighbor_vals if neighbor_val == self_value)
			if count_min is not None and count < count_min:  return DEFAULT_BAD_UTILITY
			if count_max is not None and count > count_max:  return DEFAULT_BAD_UTILITY
			result = count if weight is None else weight * count
			return result
		return utility_sameneighbor_absolute

	def builder_utility_sameneighbor_relative(
		tau_min : float | None,
		tau_max : float | None,
		weight  : float | None,
	) -> Utility_Criterion_Discrete:
		def utility_sameneighbor_relative(
			self_value    : AgentType_Value_Discrete,
			neighbor_vals : list[AgentType_Value_Discrete],
			context       : None,
		) -> float:
			count_all  = len(neighbor_vals)
			count_same = sum(1 for neighbor_val in neighbor_vals if neighbor_val == self_value)
			ratio = count_same / count_all if count_all != 0 else 0 
			# ratio = (count_same + 1) / (count_all + 1)  #TODO provide alternative ?
			if tau_min is not None and ratio < tau_min:  return DEFAULT_BAD_UTILITY
			if tau_max is not None and ratio > tau_max:  return DEFAULT_BAD_UTILITY
			result = ratio if weight is None else weight * ratio
			return result
		return utility_sameneighbor_relative

	def builder_utility_neighborinrange_absolute(
		max_dist  : float,
		count_min : int | None,
		count_max : int | None,
		weight    : float | None,
		mode      : Literal["add", "mul"],
	) -> Utility_Criterion_Continuous:
		def utility_neighborinrange_absolute(
			self_value    : AgentType_Value_Continuous,
			neighbor_vals : list[AgentType_Value_Continuous],
			context       : None,
		) -> float:
			distance = lambda x, y: abs(x - y) if mode == "add" else abs(log(x) - log(y))
			count = sum(1 for neighbor_val in neighbor_vals if distance(neighbor_val, self_value) <= max_dist)
			if count_min is not None and count < count_min:  return DEFAULT_BAD_UTILITY
			if count_max is not None and count > count_max:  return DEFAULT_BAD_UTILITY
			result = count if weight is None else weight * count
			return result
		return utility_neighborinrange_absolute


	def builder_utility_neighborinrange_relative(
		max_dist : float,
		tau_min  : float | None,
		tau_max  : float | None,
		weight   : float | None,
		mode      : Literal["add", "mul"],
	) -> Utility_Criterion_Continuous:
		def utility_neighborinrange_relative(
			self_value    : AgentType_Value_Continuous,
			neighbor_vals : list[AgentType_Value_Continuous],
			context       : None,
		) -> float:
			distance   = lambda x, y: abs(x - y) if mode == "add" else abs(log(x) - log(y))
			count_same = sum(1 for neighbor_val in neighbor_vals if distance(neighbor_val, self_value) <= max_dist)
			count_all  = len(neighbor_vals)
			ratio = count_same / count_all if count_all != 0 else 0 
			# ratio = (count_same + 1) / (count_all + 1)  #TODO provide alternative ?
			if tau_min is not None and ratio < tau_min:  return DEFAULT_BAD_UTILITY
			if tau_max is not None and ratio > tau_max:  return DEFAULT_BAD_UTILITY
			result = ratio if weight is None else weight * ratio
			return result
		return utility_neighborinrange_relative

	def builder_utility_friendandenemies(
		handle_friends : bool,
		handle_enemies : bool,
		mode           : Literal["count", "ratio"],
	) -> Utility_Criterion_AgentID:
		def utility_friendandenemies(
			self_id   : AgentID,
			neighbors : list[AgentID],
			context   : GraphType,
		) -> float:
			social_neighbors = context.neighbors(self_id)
			if isinstance(context, nx.DiGraph) or isinstance(context, nx.MultiDiGraph):
				social_relationships = context.out_edges(self_id)
			else:
				social_relationships = context.edges(self_id)
			affinities = {
				target: value
				for source, target in social_relationships
				if (value := context.get_edge_data(source, target)) is not None
			}
			count_friends = sum(
				1 for neighbor in neighbors
				if neighbor in affinities and affinities[neighbor] > 0
			)
			count_enemies = sum(
				1 for neighbor in neighbors
				if neighbor in affinities and affinities[neighbor] < 0
			)
			result = 0.0
			if mode == "count":
				if handle_friends: result += count_friends
				if handle_enemies: result -= count_enemies
			if mode == "ratio":
				result = (count_friends + 1) / (count_friends + count_enemies + 1)
			return result
		return utility_friendandenemies

	def builder_utility_specificphily(
		philys : dict[AgentType_Value_Discrete, Utility_Criterion_Discrete],
	) -> Utility_Criterion_Discrete:
		def utility_specificphily(
			self_value : AgentType_Value_Discrete,
			neighbors  : list[AgentType_Value_Discrete],
			context    : None,
		) -> float:
			if self_value not in philys:
				raise ValueError(f"Specific phily utility not defined for value '{self_value}'")
			return philys[self_value](self_value, neighbors, context)
		return utility_specificphily

	result : Utility_Criterion
	match utility_type:
		case Utility_SameNeighbor_Absolute                   (count_min, count_max, weight)           : result = builder_utility_sameneighbor_absolute(count_min, count_max, weight)
		case Utility_SameNeighbor_Relative                   (tau_min,   tau_max,   weight)           : result = builder_utility_sameneighbor_relative(tau_min, tau_max, weight)
		case Utility_NeighborInRange_Additive_Absolute       (max_dist, count_min, count_max, weight) : result = builder_utility_neighborinrange_absolute(max_dist, count_min, count_max, weight, "add")
		case Utility_NeighborInRange_Additive_Relative       (max_dist, tau_min,   tau_max,   weight) : result = builder_utility_neighborinrange_relative(max_dist, tau_min, tau_max, weight, "add")
		case Utility_NeighborInRange_Multiplicative_Absolute (log_max_dist, lc_min, lc_max, weight)   : result = builder_utility_neighborinrange_absolute(log_max_dist, lc_min, lc_max, weight, "mul")
		case Utility_NeighborInRange_Multiplicative_Relative (log_max_dist, lt_min, lt_max, weight)   : result = builder_utility_neighborinrange_relative(log_max_dist, lt_min, lt_max, weight, "mul")
		case Utility_FriendAndEnemies_Absolute               (handle_friends, handle_enemies)         : result = builder_utility_friendandenemies(handle_friends, handle_enemies, "count")
		case Utility_FriendAndEnemies_Relative               (handle_friends, handle_enemies)         : result = builder_utility_friendandenemies(handle_friends, handle_enemies, "ratio")
		case Utility_SpecificPhily                           (philys)                                 : result = builder_utility_specificphily(philys)
	return result


def get_default_utility_scalarized_function(
	domain        : AgentType_Domain,
	utility_types : dict[AgentType_Name, UtilityType] | None = None,
	combiner      : Callable[[dict[AgentType_Name, Utility_Criterion]], float] | None = None,
) -> Utility_Scalarized:
	safe_criteria_dict = {
		k: get_default_utility_criterion_function(
			Utility_SameNeighbor_Relative(None, None, None)
			if isinstance(v, list) else
			Utility_NeighborInRange_Multiplicative_Relative(0.2, None, None, None)
		)
		for k, v in domain.items()
	}
	if utility_types:
		for k, v in utility_types.items():
			safe_criteria_dict[k] = get_default_utility_criterion_function(v)

	def utility_scalarized(
		self_vector : AgentType_Vector,
		neighbors   : list[AgentType_Vector],
		context     : Any,
	) -> float:
		result = 0.0
		if combiner:
			result = combiner(safe_criteria_dict)
		else:
			for key in safe_criteria_dict.keys():
				criterion     = safe_criteria_dict[key]
				self_value    = self_vector[key]
				neighbor_vals = [neighbor[key] for neighbor in neighbors]
				partial       = criterion(self_value, neighbor_vals, context)
			#	print(f"Utility for key {key}, value {self_value}, neighbors {neighbors}, partial {partial}")
				result     += partial  # TODO sum, average, multiply for different welfares
		return result

	return utility_scalarized




"""
#######################
Agent class and utils
#######################
"""

class Agent:
	def __init__(self,
		id         : AgentID,
		value      : AgentType_Vector,
		domain     : AgentType_Domain,
		is_valid   : AgentType_Constraints,
		move_mode  : MovementMode     | None = None,
		nature     : AgentType_Nature | None = None,
		happiness  : float            | None = None,
		assign_pos : NodeID           | None = None,
		layout_pos : Position         | None = None,
	#	preferences : TODO for complex utilities ?
	):
		if not is_valid(domain, value):
			raise ValueError(f"Agent {id}'s value '{value}' does not respect constraints")
		self.id         : AgentID          = id
		self.value      : AgentType_Vector = value
		self.layout_pos : Position | None  = layout_pos
		self.graph_pos  : NodeID           = assign_pos if assign_pos is not None else -1
		self.move_mode  : MovementMode     = move_mode  if move_mode  is not None else "jump"
		self.nature     : AgentType_Nature = nature     if nature     is not None else "random"
		self.v_utility	: float            = -1
		self.happiness  : float            = happiness if happiness is not None else DEFAULT_HAPPINESS

	def get_utility_at_node(self, node: NodeID, model : SchellingModel, context : Any) -> float:
		rev_assignment        = {v: k for k, v in model.history[-1].items()}
		neighborhood          = model.topology.graph.neighbors(node)
		neighbor_agent_ids    = [rev_assignment[node_id] for node_id in neighborhood if node_id in rev_assignment]
		neighbor_agents       = [model.agents[agent_id] for agent_id in neighbor_agent_ids]
		neighbor_agent_values = [agent.value for agent in neighbor_agents]
		result                = model.utility(self.value, neighbor_agent_values, context)
		return result

	def update_utility_current(self, model: SchellingModel, context : Any) -> None:
		self.v_utility = self.get_utility_at_node(self.graph_pos, model, context)

	def get_move_jump(
		self,
		model      : SchellingModel,
		free_nodes : list[NodeID],
		mode       : Literal["any", "max"],
		context    : Any,
	) -> NodeID:
		# this function assumes that the utility value is up-to-date
		best_utility      = self.v_utility
		best_node         = self.graph_pos
		scrambled_indices = list(range(len(free_nodes)))
		while len(scrambled_indices) > 0:
			i           = random.choice(scrambled_indices)
			node        = free_nodes[i]
			new_utility = self.get_utility_at_node(node, model, context)
			if new_utility > best_utility:
				if mode == "any":
					return node
				else:
					best_utility = new_utility
					best_node    = node
			scrambled_indices.remove(i)
		return best_node

	def get_move_swap(
		self,
		model          : SchellingModel,
		occupied_nodes : list[NodeID],
		mode           : Literal["any", "max"],
		context        : Any,
	) -> NodeID:
		# this function assumes that the utility value is up-to-date
		best_utility      = self.v_utility
		best_node         = self.graph_pos
		scrambled_indices = list(range(len(occupied_nodes)))
		scrambled_indices.remove(self.graph_pos)
		while len(scrambled_indices) > 0:
			i = random.choice(scrambled_indices)
			scrambled_indices.remove(i)
			node = occupied_nodes[i]
			swap_partner = model.agents[model.history[-1][node]]
			if self.value == swap_partner.value:
				continue
			new_utility = self.get_utility_at_node(node, model, context)
			if new_utility > best_utility:
				if mode == "any":
					return node
				else:
					best_utility = new_utility
					best_node    = node
			scrambled_indices.remove(i)
		return best_node

	def get_move(
		self,
		model          : SchellingModel,
		free_nodes     : list[NodeID],
		occupied_nodes : list[NodeID],
		context        : Any,
	) -> NodeID:
		# free_nodes and occupied_nodes need to be fed as extra args
		# so that the new assignment can happen in a first-come first-serve
		# basis and avoid conflicts
		if self.nature == "stubborn" or self.v_utility >= self.happiness:
			return self.graph_pos
		elif self.nature == "random":
			if   "jump" in self.move_mode:
				return random.choice(free_nodes)
			elif "swap" in self.move_mode:
				unhappy_agents = [agent.id for agent in model.agents if agent.v_utility < agent.happiness]
				unhappy_agents.remove(self.id)
				swap_partner = random.choice(unhappy_agents)
				return swap_partner
			else:
				raise ValueError(f"Unknown move mode '{self.move_mode}'")
		elif self.nature == "strategic":
			if   self.move_mode == "jump"     :  return self.get_move_jump(model, free_nodes,     "any", context)
			elif self.move_mode == "swap"     :  return self.get_move_swap(model, occupied_nodes, "any", context)
			elif self.move_mode == "max_jump" :  return self.get_move_jump(model, free_nodes,     "max", context)
			elif self.move_mode == "max_swap" :  return self.get_move_swap(model, occupied_nodes, "max", context)
			else:
				raise ValueError(f"Unknown move mode '{self.move_mode}'")
		else:
			raise ValueError(f"Unknown agent nature '{self.nature}'")

		


"""
#######################
Model types and utils
#######################
"""

#TODO provide trackability for utility criteria when possible, not just scalarized (?)
@dataclass
class SchellingModelConfig_Explicit:
	topology    : Topology
	agents      : list[Agent]
	domain      : AgentType_Domain
	move_mode   : MovementMode | None
	constraints : AgentType_Constraints | None
	utility     : Utility_Scalarized | None
	assignment  : Assignment | None
	max_iter    : int
	colormap    : AgentType_ColorMap | None
	social_net  : GraphType | None

@dataclass
class SchellingModelConfig_Random:
	topology      : tuple[GraphType_Literal, TopologyType]
	n_agents      : int
	agent_natures : AgentType_NatureProportions | None
	happiness     : float | None
	domain        : AgentType_Domain
	move_mode     : MovementMode | None
	constraints   : AgentType_Constraints | None
	utility       : Utility_Scalarized | None
	distributions : AgentType_Distributions | None  # can be partially defined, the rest of the domain will default
	max_iter      : int
	colormap      : AgentType_ColorMap | None
	social_net    : GraphType | None

SchellingModelConfig = Union[
	SchellingModelConfig_Explicit,
	SchellingModelConfig_Random,
]

class SchellingModel:
	def __init__(self, config: SchellingModelConfig):
		self.topology   : Topology
		self.domain     : AgentType_Domain
		self.move_mode  : MovementMode
		self.is_valid   : AgentType_Constraints
		self.utility    : Utility_Scalarized
		self.agents     : list[Agent]
		self.max_iter   : int
		self.history    : list[Assignment]
		self.colormap   : AgentType_ColorMap
		self.social_net : GraphType | None
		self.figures	: dict[AgentType_Name, dict[int, plt.Figure]]
		match config:
			case SchellingModelConfig_Explicit(
				topology,
				agents,
				domain,
				move_mode,
				constraints,
				utility,
				assignment,
				max_iter,
				colormap,
				social_net,
			):
				self.topology   = topology
				self.domain     = domain
				self.move_mode  = move_mode if move_mode is not None else "jump"
				self.is_valid   = constraints if constraints is not None else SchellingModel.get_is_valid(domain)
				self.max_iter   = max_iter
				self.utility	= utility if utility is not None else get_default_utility_scalarized_function(domain)
				self.history    = [assignment] if assignment is not None else [self.get_random_assignment(len(agents))]
				self.agents     = agents
				self.colormap   = get_default_colormap(domain, colormap)
				self.social_net = social_net
			case SchellingModelConfig_Random(
				topology,
				n_agents,
				agent_natures,
				happiness_threshold,
				domain,
				move_mode,
				constraints,
				utility,
				distributions,
				max_iter,
				colormap,
				social_net,
			):
				self.topology   = Topology(TopologyConfig_Generated(*topology))
				self.domain     = domain
				self.move_mode  = move_mode if move_mode is not None else "jump"
				self.is_valid   = constraints if constraints is not None else SchellingModel.get_is_valid(domain)
				self.max_iter   = max_iter
				self.utility    = utility if utility is not None else get_default_utility_scalarized_function(domain)
				self.history    = [self.get_random_assignment(n_agents)]
				self.agents     = self.generate_agents(n_agents, distributions, agent_natures, happiness_threshold)
				self.colormap   = get_default_colormap(domain, colormap)
				self.social_net = social_net
			case _:
				raise ValueError("Invalid SchellingModelConfig")
		if len(self.agents) > len(self.topology.graph.nodes()):
			raise ValueError("SchellingModel.__init__(): Not enough nodes for all agents")
		self.equilibrium_found = False
		self.update_agents_with_assignment(self.history[-1])
		self.figures = {}


	@staticmethod
	def get_is_valid(
		domain            : AgentType_Domain,
		extra_constraints : list[AgentType_VectorConstraint] | None = None,
	) -> AgentType_Constraints:
		def discrete_lambda_builder(bounds: list[AgentType_Value_Discrete]) -> AgentType_ValueConstraint:
			def is_criterion_valid_discrete(s: AgentType_Value_Discrete) -> bool:
				return s in bounds
			return is_criterion_valid_discrete

		def continuous_lambda_builder(bounds: tuple[AgentType_Value_Continuous, AgentType_Value_Continuous]) -> AgentType_ValueConstraint:
			def is_criterion_valid_continuous(x: AgentType_Value_Continuous) -> bool:
				return bounds[0] <= x and x <= bounds[1]
			return is_criterion_valid_continuous

		lambdas : dict[AgentType_Name, AgentType_ValueConstraint] = {}
		for type_name, bounds in domain.items():
			if   isinstance(bounds, list  ):  lambdas[type_name] = discrete_lambda_builder(bounds)
			elif isinstance(bounds, tuple ):  lambdas[type_name] = continuous_lambda_builder(bounds)
			else:
				raise ValueError(f"Invalid domain value '{bounds}'")

		def is_valid(domain: AgentType_Domain, value: AgentType_Vector) -> bool:
			for type_name, subvalue in value.items():
				constraint = lambdas[type_name]
				if not constraint(subvalue):
					print(f"Failed constraint {constraint} for '{type_name}' with value '{subvalue}'")
					return False
			if extra_constraints:
				for constraint in extra_constraints:
					if not constraint(value):
						print(f"Failed extra constraint {constraint}")
						return False
			return True

		return is_valid

	def generate_agents(
		self,
		n             : int,
		distributions : AgentType_Distributions     | None = None,
		agent_natures : AgentType_NatureProportions | None = None,
		happiness     : float                       | None = None,
	) -> list[Agent]:
		def match_default_distribution(v: AgentType_Categories | AgentType_Range) -> DistributionGenerator:
			if   isinstance(v, list ):  result = lambda : random.choice(v)
			elif isinstance(v, tuple):  result = lambda : random.uniform(v[0], v[1])
			else:
				raise ValueError(f"match_default_distribution(): Invalid domain value '{v}'")
			return result

		def setup_distributions(
			distributions : AgentType_Distributions | None,
		) -> DomainDistributions:
			safe_distributions : DomainDistributions = {
				k: match_default_distribution(v)
				for k, v in self.domain.items()
			}
			if distributions:
				for k, v in distributions.items():
					if k not in self.domain:
						raise ValueError(f"setup_distributions(): Invalid distribution key '{k}'")
					safe_distributions[k] = distribution_type_to_generator(v)
			return safe_distributions

		result : list[Agent] = []
		safe_distributions = setup_distributions(distributions)
		natures : list[AgentType_Nature]
		if agent_natures:
			natures = random.choices( # type:ignore
				list(agent_natures.keys()), # type:ignore
				weights = list(agent_natures.values()), # type:ignore
				k       = n,
			)
		else:
			natures = ["random"] * n
		for i in range(n):
			value : AgentType_Vector = { k: v() for k, v in safe_distributions.items() }
			agent = Agent(i, value, self.domain, self.is_valid, self.move_mode, natures[i], happiness=happiness)
			result.append(agent)
		return result

	def get_random_assignment(self, agent_amount : int) -> Assignment:
		result : Assignment = {}
		available_nodes = list(self.topology.graph.nodes())
		for agent_id in range(agent_amount):
			node = random.choice(available_nodes)
			available_nodes.remove(node)
			result[agent_id] = node
		return result

	def update_agents_with_assignment(self, assignment: Assignment) -> None:
		for agent in self.agents:
			agent.graph_pos = assignment[agent.id]

	def get_free_nodes(self) -> list[NodeID]:
		occupied = self.history[-1].values()
		result : list[NodeID] = [  # type:ignore
			node
			for node in self.topology.graph.nodes()
			if node not in occupied
		]
		return result

	def get_occupied_nodes(self) -> list[NodeID]:
		result = list(self.history[-1].values())
		return result

	def get_next_step(self) -> Assignment:
		current_state  = self.history[-1].copy()
		free_nodes     = self.get_free_nodes()
		occupied_nodes = self.get_occupied_nodes()
		new_assignment = {}
		if self.move_mode == "jump" or self.move_mode == "max_jump":
			for agent in self.agents:
				new_node = agent.get_move(self, free_nodes, [], self.social_net)
				new_assignment[agent.id] = new_node
				if new_node != agent.graph_pos:
					free_nodes.remove(new_node)
				if len(free_nodes) == 0:
					break
		elif self.move_mode == "swap" or self.move_mode == "max_swap":
			for agent in self.agents:
				new_node = agent.get_move(self, [], occupied_nodes, self.social_net)
				swap_partner = self.agents[current_state[new_node]]
				new_assignment[agent.id] = new_node
				new_assignment[swap_partner.id] = agent.graph_pos
				if new_node != agent.graph_pos:
					occupied_nodes.remove(new_node)
					occupied_nodes.remove(agent.graph_pos)
				if len(occupied_nodes) < 2:
					break
		current_state.update(new_assignment)
		result = current_state
		return result

	def run(self) -> None:
		i = 0
		while i < self.max_iter:
			print(f"\033[94mIteration {i}\033[0m")
			next_step = self.get_next_step()
			if next_step == self.history[-1]:
				self.equilibrium_found = True
				self.max_iter = i
			self.history.append(next_step)
			self.update_agents_with_assignment(next_step)
			print(f"Utilities: {[f'{float(agent.v_utility):.2}' for agent in self.agents]}")
			for agent in self.agents:
				agent.update_utility_current(self, self.social_net)
			i += 1
		print(f"Run done after {self.max_iter}")
		print(f"Equilibrium found ? {self.equilibrium_found}")

	def get_figure(
		self,
		iter_step   : int,
		type_name   : AgentType_Name,
	#	type_mode   : Literal["type", "utility"],
		nodes_pos   : NodePosDict | None,
		with_labels : bool,
		with_edges  : bool,
	) -> plt.Figure:
		# https://networkx.org/documentation/stable/reference/drawing.html
		if type_name in self.figures:
			if iter_step in self.figures[type_name]:
				return self.figures[type_name][iter_step]
		else:
			self.figures[type_name] = {}
		graph = self.topology.graph.copy(as_view=False)
		if not nodes_pos:
			nodes_pos = self.topology.get_layout(graph)
		labels = {
			node_id : self.history[iter_step][node_id] if node_id in self.history[iter_step] else ""
			for node_id in graph.nodes()
		}

		# Handle colormapping
		#TODO if type_mode == "type":
		values = {}
		colors = {}
		step_assignment = self.history[iter_step]
		rev_assignment  = {v: k for k, v in step_assignment.items()}
		colormap = self.colormap[type_name]
		for node_id in rev_assignment:
			agent_id        = rev_assignment[cast(int, node_id)]
			agent_value     = self.agents[agent_id].value[type_name]
			values[node_id] = agent_value
		type_is_discrete = isinstance(colormap, dict)
		max_value : float = max(values.values()) if not type_is_discrete else 0.0  #type:ignore
		min_value : float = min(values.values()) if not type_is_discrete else 0.0  #type:ignore
		norm_value = max_value - min_value
		for node_id in graph.nodes():
			if node_id not in values:
				colors[node_id] = "white"
				continue
			elif type_is_discrete:
				agent_value     = values[node_id]
				colors[node_id] = colormap[agent_value]  # type:ignore
			else:
				agent_value            = values[node_id] 
				normalized_agent_value = (agent_value - min_value) / norm_value # type:ignore
				colors[node_id]        = colormap(normalized_agent_value)  # type:ignore

		fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE, dpi=DEFAULT_DPI)
		nx.draw(
			self.topology.graph,
			nodes_pos, 
			ax          = ax,
			with_labels = with_labels,
			labels      = labels,
			node_color  = list(colors.values()),
			edge_color  = "gray" if with_edges else "white",
			edgecolors  = "gray",
			node_size   = 50,
			font_size   = 5,
		)
		ax.set_title(f"{type_name}")
		self.figures[type_name][iter_step] = fig
		return fig

	# TODO
	def compute_metrics(self) -> dict:
		raise NotImplementedError("SchellingModel.compute_metrics(): Not implemented")




"""
#######################
Kivy App UI
types and utils 
#######################
"""

class MainWindow(BoxLayout):
	def __init__(self, model: SchellingModel, **kwargs):
		super().__init__(**kwargs)
		self.orientation = 'horizontal'
		self.model = model
		self.model.run()
		self.graph_layout = None
		self.with_labels  = False
		self.with_edges   = False
		# TODO show color legend
		self.build_controls()
		self.render()

	def render(self) -> None:
		if self.graph_layout:
			self.remove_widget(self.graph_layout)
		self.graph_layout = self.create_graph_canvas()
		self.add_widget(self.graph_layout)

	# Export utils

	@staticmethod
	def get_export_filename(iter_step: int, type_name: AgentType_Name) -> str:
		return f"{TMP_IMG_DIR}{sep}schelling_{type_name}_{iter_step}.png"

	def export_png_plot(self, fig: plt.Figure, filename: str) -> None:
		fig.set_size_inches(DEFAULT_FIGSIZE, forward=True)
		fig.savefig(f"{OUT_IMG_DIR}{sep}{filename}", bbox_inches='tight', pad_inches = DEFAULT_PADDING)

	def export_png_plot_at_iter(self, iter_step: int, type_name: AgentType_Name) -> str:
		fig = self.model.get_figure(
			iter_step   = iter_step,
			type_name   = type_name,
			nodes_pos   = None,
			with_labels = self.with_labels,
			with_edges  = self.with_edges,
		)
		fig_name = MainWindow.get_export_filename(iter_step, type_name)
		self.export_png_plot(fig, fig_name)
		return fig_name

	def export_png_all_plots_at_iter(self, iter_step: int) -> str:
		fig_paths = []
		for type_name in self.model.domain.keys():
			fig_path = self.export_png_plot_at_iter(iter_step, type_name)
			fig_paths.append(fig_path)
		# combine figures into a single plot
		result_fig = combine_img_plots(fig_paths)
		fig_name = f"{TMP_IMG_DIR}{sep}schelling_all_{iter_step}.png"
		result_fig.savefig(fig_name, bbox_inches='tight', pad_inches = DEFAULT_PADDING)
		return fig_name

	def export_gif_plot(self, type_name: AgentType_Name) -> str:
		fig_paths = []
		for iter_step in range(self.model.max_iter):
			print(f"Building {iter_step} image for gif...")
			sleep(0.01)
			fig_path = self.export_png_plot_at_iter(iter_step, type_name)
			fig_paths.append(fig_path)
		gif_name = f"schelling_{type_name}.gif"
		export_gif_from_pngs(fig_paths, gif_name)
		return gif_name

	def export_gif_all_plots(self) -> str:
		fig_paths = []
		for iter_step in range(self.model.max_iter):
			print(f"Building {iter_step} images for gif...")
			sleep(0.01)
			fig_path = self.export_png_all_plots_at_iter(iter_step)
			fig_paths.append(fig_path)
		gif_name = f"schelling_all.gif"
		export_gif_from_pngs(fig_paths, gif_name)
		return gif_name

	# Event Bindings

	def on_export_png(self, instance) -> None:
		iter_step = self.get_iter_step()
		type_name = self.get_selected_type()
		self.export_png_plot_at_iter(iter_step, type_name)

	def on_export_png_all(self, instance) -> None:
		fig_paths = []
		for type_name in self.model.domain.keys():
			fig = self.model.get_figure(
				iter_step   = self.get_iter_step(),
				type_name   = type_name,
				nodes_pos   = None,
				with_labels = self.with_labels,
				with_edges  = self.with_edges,
			)
			fig_name = f"{TMP_IMG_DIR}{sep}schelling_{type_name}.png"
			fig.savefig(fig_name, bbox_inches='tight', pad_inches = DEFAULT_PADDING)
			fig_paths.append(fig_name)
		# combine figures into a single plot
		result_fig = combine_img_plots(fig_paths)
		result_fig.savefig(f"schelling_all.png", bbox_inches='tight', pad_inches = DEFAULT_PADDING)

	def on_export_gif(self, instance) -> None:
		type_name = self.get_selected_type()
		self.export_gif_plot(type_name)

	def on_export_gif_all(self, instance) -> None:
		self.export_gif_all_plots()



	def on_iter_value_change(self, instance, value):
		self.slider_ilabel.text = f"Iteration step: {int(value)}"
		self.iter_value = int(value)
		self.render()

	def on_toggle_labels(self, instance):
		self.with_labels        = not self.with_labels
		self.toggle_labels.text = "Show labels" if not self.with_labels else "Hide labels"
		self.model.figures      = {}  # TODO better way to force rerender through cache invalidation
		self.render()

	def on_toggle_edges(self, instance):
		self.with_edges        = not self.with_edges
		self.toggle_edges.text = "Show edges"   if not self.with_edges  else "Hide edges"
		self.model.figures     = {}  # TODO better way to force rerender through cache invalidation
		self.render()

	def on_type_radio_selected(self, instance):
		instance.state = "down"
		self.render()

	# Dashboard builders

	@staticmethod
	def build_radiobutton(
		labels     : Iterable[str],
		group_name : str,
		binding    : Callable[[ToggleButton], None],
	) -> BoxLayout:
		radiobutton_layout = BoxLayout(orientation='horizontal')
		buttons = [
			ToggleButton(
				text  = label,
				group = group_name,
				state = ("down" if i == 0 else "normal"),
			)
			for i, label in enumerate(labels)
		]
		for button in buttons:
			button.bind(on_press = binding)
			radiobutton_layout.add_widget(button)
		return radiobutton_layout


	def build_export_buttons(self) -> BoxLayout:
		export_layout  = BoxLayout(orientation='horizontal')
		export_png     = Button(text="Export PNG")
		export_png_all = Button(text="Export PNG all")
		export_gif     = Button(text="Export GIF")
		export_gif_all = Button(text="Export GIF all")
		export_png     .bind(on_press = self.on_export_png)
		export_png_all .bind(on_press = self.on_export_png_all)
		export_gif     .bind(on_press = self.on_export_gif)
		export_gif_all .bind(on_press = self.on_export_gif_all)
		export_layout.add_widget(export_png)
		export_layout.add_widget(export_png_all)
		export_layout.add_widget(export_gif)
		export_layout.add_widget(export_gif_all)
		return export_layout


	def build_toggles(self) -> BoxLayout:
		toggles_layout     = BoxLayout(orientation='horizontal')
		self.toggle_edges  = ToggleButton(text = "Show edges"  if not self.with_edges  else "Hide edges"  )
		self.toggle_labels = ToggleButton(text = "Show labels" if not self.with_labels else "Hide labels" )
		self.toggle_edges  .bind(on_press = self.on_toggle_edges)
		self.toggle_labels .bind(on_press = self.on_toggle_labels)
		toggles_layout.add_widget(self.toggle_edges)
		toggles_layout.add_widget(self.toggle_labels)
		return toggles_layout

	def build_iterator_slider(self) -> BoxLayout:
		iter_slider_layout = BoxLayout(orientation='vertical')
		self.slider_ilabel = Label(text="Iteration step: 0")
		self.slider_iters  = Slider(min = 0, max = self.model.max_iter, value = 0)
		self.slider_iters  .bind(value = self.on_iter_value_change)
		iter_slider_layout.add_widget(self.slider_iters)
		iter_slider_layout.add_widget(self.slider_ilabel)
		return iter_slider_layout

	def build_controls(self) -> None:
		controls_layout    = BoxLayout(orientation='vertical')
		self.iter_slider   = self.build_iterator_slider()
		self.type_radio	   = MainWindow.build_radiobutton(self.model.domain.keys(), "type_radio", self.on_type_radio_selected)
		self.toggles       = self.build_toggles()
		self.exports       = self.build_export_buttons()
		controls_layout.add_widget(self.iter_slider)
		controls_layout.add_widget(self.type_radio)
		controls_layout.add_widget(self.toggles)
		controls_layout.add_widget(self.exports)
		self.add_widget(controls_layout)


	def create_graph_canvas(self) -> FigureCanvasKivyAgg:
		fig = self.model.get_figure(
			iter_step   = self.get_iter_step(),
			type_name   = self.get_selected_type(),
			nodes_pos   = None,
			with_labels = self.with_labels,
			with_edges  = self.with_edges,
		)
		result           = FigureCanvasKivyAgg(fig)
		return result

	def get_iter_step(self) -> int:
		result = int(self.slider_iters.value)
		return result

	def get_selected_type(self) -> AgentType_Name:
		result = [button.text for button in self.type_radio.children if button.state == "down"][0]
		return result


class SchellingApp(App):
	def __init__(self, model: SchellingModel, **kwargs):
		super().__init__(**kwargs)
		self.model = model

	def build(self):
		return MainWindow(self.model)



"""
#######################
Model Config examples
#######################
"""

def example_simple_game() -> SchellingModel:
	MAX_ITERATIONS    = 200
	GRID_TOPO_DIM     = (20, 20)
	AGENT_AMOUNT      = 300
	DEFAULT_HAPPINESS = 0.5
	DOMAIN_RACE_RELIGION_INCOME : AgentType_Domain = {
		"race"     : ["white", "black"],
	}
	COLORMAP_RACE_RELIGION : AgentType_ColorMap = {
		"race" : {
			"white" : "pink",
			"black" : "brown",
		},
	}
	DISTRIBUTIONS_RACE_RELIGION_INCOME : AgentType_Distributions = {
		"race"     : Distribution_Choice_Categories(["white", "black"]),
	}
	AGENT_NATURES : AgentType_NatureProportions = {
		"random"   : 0.,
		"stubborn" : 0.,
		"strategic": 1.,
	}
	CONSTRAINTS_RACE_RELIGION_INCOME : AgentType_Constraints | None = None
	UTILITY_RACE_RELIGION_INCOME     : Utility_Scalarized    | None = None
	model_config = SchellingModelConfig_Random(
		topology      = ("Graph", Topology_GridDiagonals(GRID_TOPO_DIM)),
		n_agents      = AGENT_AMOUNT,
		move_mode     = "jump",
		agent_natures = AGENT_NATURES,
		happiness     = DEFAULT_HAPPINESS,
		domain        = DOMAIN_RACE_RELIGION_INCOME,
		constraints   = CONSTRAINTS_RACE_RELIGION_INCOME,
		utility       = UTILITY_RACE_RELIGION_INCOME,
		distributions = DISTRIBUTIONS_RACE_RELIGION_INCOME,
		max_iter      = MAX_ITERATIONS,
		colormap      = COLORMAP_RACE_RELIGION,
		social_net    = None,
	)
	return SchellingModel(model_config)


def example_complex_game() -> SchellingModel:
	MAX_ITERATIONS    = 200
	GRID_TOPO_DIM     = (20, 20)  # (40, 40)
	AGENT_AMOUNT      = 350       # 1500
	DEFAULT_HAPPINESS = 1.5
	DOMAIN_RACE_RELIGION_INCOME : AgentType_Domain = {
		"race"     : ["white", "black", "asian"],
		"religion" : ["christian", "muslim", "jewish"],
		"income"   : (0, 10000000),
	}
	COLORMAP_RACE_RELIGION : AgentType_ColorMap = {
		"race" : {
			"white" : "pink",
			"black" : "brown",
			"asian" : "orange",
		},
		"religion": {
			"christian" : "red",
			"muslim"    : "green",
			"jewish"    : "blue",
		}
	}
	DISTRIBUTIONS_RACE_RELIGION_INCOME : AgentType_Distributions = {
		"race"     : Distribution_Choice_Categories(["white", "black", "asian"]),
		"religion" : Distribution_CustomDiscrete({
			"christian" : 0.6,
			"muslim"    : 0.3,
			"jewish"    : 0.1,
		}),
		#"income" : Distribution_UniformContinuous(0, 10000000),
	}
	AGENT_NATURES : AgentType_NatureProportions = {
		"random"   : 0.,
		"stubborn" : 0.,
		"strategic": 1.,
	}
	CONSTRAINTS_RACE_RELIGION_INCOME : AgentType_Constraints | None = None

	def same_race_and_religion_above_all(
		self_value : AgentType_Vector,
		neighbor_values : list[AgentType_Vector],
		context : None,
	) -> float:
		neighbor_values = [v for v in neighbor_values]
		count_all = len(neighbor_values)
		count_both_same = sum(
			1 for neighbor_value in neighbor_values
			if neighbor_value["race"] == self_value["race"]
			and neighbor_value["religion"] == self_value["religion"]
		)
		ratio_same = count_both_same / count_all if count_all != 0 else 0
		count_similar = sum(
			0.3 for neighbor in neighbor_values
			if neighbor["race"] == self_value["race"]
			and neighbor["religion"] == self_value["religion"]
		)
		ratio_similar = count_similar / count_all if count_all != 0 else 0
		neighbors_with_similar_income = sum([
			1 for neighbor_value in neighbor_values
			if abs(log(neighbor_value["income"]) - log(self_value["income"])) <= 0.2  #type:ignore
		])
		income_friendliness_bonus = 0.5 if neighbors_with_similar_income > 0 else 0.0
		result = ratio_same + ratio_similar + income_friendliness_bonus
		return result
			

	UTILITY_RACE_RELIGION_INCOME : Utility_Scalarized = same_race_and_religion_above_all
	model_config = SchellingModelConfig_Random(
		topology      = ("Graph", Topology_GridDiagonals(GRID_TOPO_DIM)),
		n_agents      = AGENT_AMOUNT,
		move_mode     = "jump",
		agent_natures = AGENT_NATURES,
		happiness     = DEFAULT_HAPPINESS,
		domain        = DOMAIN_RACE_RELIGION_INCOME,
		constraints   = CONSTRAINTS_RACE_RELIGION_INCOME,
		utility       = UTILITY_RACE_RELIGION_INCOME,
		distributions = DISTRIBUTIONS_RACE_RELIGION_INCOME,
		max_iter      = MAX_ITERATIONS,
		colormap      = COLORMAP_RACE_RELIGION,
		social_net    = None,
	)
	return SchellingModel(model_config)




"""
#######################
Miscellaneous utils
#######################
"""

def combine_img_plots(fig_paths: list[str]) -> plt.Figure:
	result_fig, axes = plt.subplots(1, len(fig_paths))
	for ax, fig_path in zip(axes, fig_paths):
		img = plt.imread(fig_path)
		ax.imshow(img)
		ax.axis('off')
	return result_fig

def export_gif_from_pngs(
	png_paths    : list[str],
	gif_filename : str,
	duration     : int | None = None
) -> None:
	duration = duration if duration is not None else DEFAULT_GIF_FRAME_DURATION
	images = [Image.open(png) for png in png_paths]
	images[0].save(
		gif_filename, 
		save_all      = True, 
		append_images = images[1:], 
		duration      = duration,
	)




"""
#######################
Main and some constants
#######################
"""

if __name__ == '__main__':
	OUT_IMG_DIR = "out_images"
	if not exists(OUT_IMG_DIR):
		mkdir(OUT_IMG_DIR)
	TMP_IMG_DIR = f"out_images{sep}tmp_images"
	if not exists(TMP_IMG_DIR):
		mkdir(TMP_IMG_DIR)

	DEFAULT_PADDING = 0.1
	DEFAULT_FIGSIZE = (5, 5)
	DEFAULT_DPI     = 100
	DEFAULT_GIF_FRAME_DURATION = 100

	DEFAULT_HAPPINESS   = 0.5
	DEFAULT_BAD_UTILITY = -1.0

	SEED = 0
	random.seed(0)
	model1 = example_simple_game()
	model2 = example_complex_game()
	SchellingApp(model2).run()
