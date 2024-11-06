from __future__ import annotations
from typing import Any, Literal, TYPE_CHECKING
import random

from src.types import (
	Position,
	NodeID,
	AgentID,
	AgentType_Vector,
	AgentType_Domain,
	AgentType_Constraints,
	AgentType_Nature,
	MovementMode,
)
from src.config_defaults import DEFAULT_HAPPINESS
if TYPE_CHECKING:
	from src.model import SchellingModel



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
