from __future__ import annotations
from typing import Union, Iterable, cast
from dataclasses import dataclass
from threading import Thread, Lock
import random

from matplotlib.pyplot import Figure, subplots
from networkx          import draw

from src.types import (
	GraphType,
	GraphType_Literal,
	NodeID,
	NodePosDict,
	AgentType_Name,
	AgentType_Value_Discrete,
	AgentType_Value_Continuous,
	AgentType_Vector,
	AgentType_ValueConstraint,
	AgentType_VectorConstraint,
	AgentType_Domain,
	AgentType_Constraints,
	AgentType_ColorMap,
	AgentType_NatureProportions,
	AgentType_Categories,
	AgentType_Range,
	AgentType_Nature,
	Assignment,
	Utility_Scalarized,
	MovementMode,
	DomainFigureHistories,
	ConfiguredFigureHistories,
	ConfigedFigureHistories_Key,
)
from src.topology import (
	Topology,
	TopologyType,
	TopologyConfig_Generated,
)
from src.distributions import (
	AgentType_Distributions,
	DistributionGenerator,
	DomainDistributions,
	distribution_type_to_generator,
)
from src.agent           import Agent
from src.utility         import get_default_utility_scalarized_function
from src.colors          import get_default_colormap
from src.config_defaults import DEFAULT_FIGSIZE, DEFAULT_DPI



GRAPH_HISTORY_MUTEX = Lock()

#TODO provide trackability for utility criteria when possible, not just scalarized (?)
@dataclass
class SchellingModelConfig_Explicit:
	topology    : Topology
	agents      : list[Agent]
	domain      : AgentType_Domain
	move_mode   : MovementMode                | None
	constraints : AgentType_Constraints       | None
	utility     : Utility_Scalarized          | None
	assignment  : Assignment                  | None
	max_iter    : int
	colormap    : AgentType_ColorMap          | None
	social_net  : GraphType                   | None

@dataclass
class SchellingModelConfig_Random:
	topology      : tuple[GraphType_Literal, TopologyType]
	n_agents      : int
	agent_natures : AgentType_NatureProportions | None
	happiness     : float                       | None
	domain        : AgentType_Domain
	move_mode     : MovementMode                | None
	constraints   : AgentType_Constraints       | None
	utility       : Utility_Scalarized          | None
	distributions : AgentType_Distributions     | None  # can be partially defined, the rest of the domain will default
	max_iter      : int
	colormap      : AgentType_ColorMap          | None
	social_net    : GraphType                   | None

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
		self.figures	: ConfiguredFigureHistories
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
		self.figures = {
			"N_edge_N_label" : {},
			"N_edge_Y_label" : {},
			"Y_edge_N_label" : {},
			"Y_edge_Y_label" : {},
		}


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
	) -> Figure:
		# https://networkx.org/documentation/stable/reference/drawing.html
		config_key : ConfigedFigureHistories_Key = f"{'N' if with_edges else 'Y'}_edge_{'N' if with_labels else 'Y'}_label"  # type:ignore
		if type_name in self.figures[config_key]:
			if iter_step in self.figures[config_key][type_name]:
				return self.figures[config_key][type_name][iter_step]
		else:
			self.figures[config_key][type_name] = {}
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

		fig, ax = subplots(figsize = DEFAULT_FIGSIZE, dpi = DEFAULT_DPI)
		draw(
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
		GRAPH_HISTORY_MUTEX.acquire()
		self.figures[config_key][type_name][iter_step] = fig
		GRAPH_HISTORY_MUTEX.release()
		return fig

	def build_all_figures_from_config(
		self,
		type_names  : Iterable[AgentType_Name],
		with_labels : bool,
		with_edges  : bool,
	) -> None:
		threads = []
		for iter_step in range(self.max_iter):
			for type_name in type_names:
				def draw_graph():
					print(f"Starting thread for {type_name} at iteration {iter_step}")
					self.get_figure(
						iter_step   = iter_step,
						type_name   = type_name,
						nodes_pos   = None,
						with_labels = with_labels,
						with_edges  = with_edges,
					)
					print(f"Completed thread for {type_name} at iteration {iter_step}")
				graph_thread = Thread(target=draw_graph, args=())
				graph_thread.start()
				threads.append(graph_thread)
		for thread in threads:
			thread.join()


	def get_figure_history_from_config(
		self,
		with_labels : bool,
		with_edges  : bool,
	) -> DomainFigureHistories:
		config_key : ConfigedFigureHistories_Key = f"{'N' if with_edges else 'Y'}_edge_{'N' if with_labels else 'Y'}_label"  # type:ignore
		return self.figures[config_key]

	# TODO
	def compute_metrics(self) -> dict:
		raise NotImplementedError("SchellingModel.compute_metrics(): Not implemented")

