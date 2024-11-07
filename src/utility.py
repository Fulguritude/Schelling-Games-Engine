from __future__ import annotations
from typing import Callable, Literal, Any
from math import log

from networkx import DiGraph, MultiDiGraph

from src.types import (
	GraphType,
	AgentID,
	AgentType_Name,
	AgentType_Value_Discrete,
	AgentType_Value_Continuous,
	AgentType_Vector,
	AgentType_Domain,
	Utility_Criterion,
	Utility_Scalarized,
	Utility_Criterion_AgentID,
	Utility_Criterion_Discrete,
	Utility_Criterion_Continuous,
	GenericAgentValue,
)
from src.config_defaults import DEFAULT_BAD_UTILITY




def any_sum(
	self_value    : GenericAgentValue,
	neighbor_vals : list[GenericAgentValue],
	condition     : Callable[[GenericAgentValue, GenericAgentValue], bool],
	mapping_func  : Callable[[GenericAgentValue, GenericAgentValue], float],
) -> float:
	"""
	Counts the number of neighbors of a central agent that satisfy a
	condition against that central agent, maps them to a specific value,
	then sums that into a float.
	"""
	count = sum(
		mapping_func(self_value, neighbor_val)
		for neighbor_val in neighbor_vals
		if condition(self_value, neighbor_val)
	)
	return count

def simple_sum(
	self_value    : GenericAgentValue,
	neighbor_vals : list[GenericAgentValue],
	condition     : Callable[[GenericAgentValue, GenericAgentValue], bool],
) -> float:
	"""
	Counts the number of neighbors of a central agent that satisfy a condition
	"""
	count = any_sum(self_value, neighbor_vals, condition, lambda x, y: 1.0)
	return count


def ratio_func_basic(
	count_same : int | float,
	count_all  : int | float,
) -> float:
	"""
	Returns the ratio of similar neighbors to all neighbors.
	If the individual is alone, they are unhappy.
	"""
	ratio = count_same / count_all if count_all != 0 else 0 
	return ratio

def ratio_func_modified(
	count_same : int | float,
	count_all  : int | float,
) -> float:
	"""
	Returns the ratio of similar neighbors to all neighbors.
	If the individual is alone, they are happy.
	"""
	ratio = (count_same + 1) / (count_all + 1)
	return ratio

def builder_condition_in_range(
	distance_func : Callable[[GenericAgentValue, GenericAgentValue], float],
	max_dist      : float,
) -> Callable[[GenericAgentValue, GenericAgentValue], bool]:
	def condition(
		self_value    : GenericAgentValue,
		neighbor_val  : GenericAgentValue,
	) -> bool:
		distance = distance_func(self_value, neighbor_val)
		result   = distance <= max_dist
		return result
	return condition

def distance_absolute(
	self_value   : AgentType_Value_Continuous,
	neighbor_val : AgentType_Value_Continuous,
) -> float:
	distance = abs(self_value - neighbor_val)
	return distance

def distance_logarithmic(
	self_value   : AgentType_Value_Continuous,
	neighbor_val : AgentType_Value_Continuous,
) -> float:
	distance = abs(log(self_value) - log(neighbor_val))
	return distance


def builder_utility_similarneighbor(
	condition       : Callable[[GenericAgentValue, GenericAgentValue], bool],
	summand_mapping : Callable[[GenericAgentValue, GenericAgentValue], float],
	ratio_func      : Callable[[int | float, int | float], float] | None,
	weight          : float | None,
) -> Utility_Criterion:
	"""
	Builds a utility function that returns a float from the number of neighbors
	with similar values. I.e.: a builder for mono-criterion utility functions.

	Parameters:
	- condition: a function that takes a "central" agent and "neighbor"'s values and
	returns a boolean, to know whether the neighbor is similar to the central agent.
	- summand_mapping: a function that takes the central agent's value and the neighbor's
	and maps is somehow, can be used for specific weighing
	- ratio_func: a function that takes the number of similar neighbors vs all neighbors
	(used to handle zero or whether to count the central agent itself)
	- weight: a float that can be used to multiply the result with, in order to better
	manipulate multi-criteria utility
	"""
	if ratio_func is None:
		def utility_sameneighbor_absolute(
			self_value    : GenericAgentValue,
			neighbor_vals : list[GenericAgentValue],
			context       : None,
		) -> float:
			count  = any_sum(self_value, neighbor_vals, condition, summand_mapping)
			result = count if weight is None else weight * count
			return result
		result = utility_sameneighbor_absolute

	else:
		def utility_sameneighbor_relative(
			self_value    : GenericAgentValue,
			neighbor_vals : list[GenericAgentValue],
			context       : None,
		) -> float:
			count_all  = len(neighbor_vals)
			count_same = any_sum(self_value, neighbor_vals, condition, summand_mapping)
			ratio      = ratio_func(count_same, count_all)
			result     = ratio if weight is None else weight * ratio
			return result
		result = utility_sameneighbor_relative
	return result

def builder_utility_neighborinrange(
	ratio_func      : Callable[[int | float, int | float], float] | None,
	summand_mapping : Callable[[AgentType_Value_Continuous, AgentType_Value_Continuous], float],
	distance_func   : Callable[[AgentType_Value_Continuous, AgentType_Value_Continuous], float],
	max_dist        : float,
	weight          : float | None,
) -> Utility_Criterion_Continuous:
	condition = builder_condition_in_range(distance_func, max_dist)
	result = builder_utility_similarneighbor(condition, summand_mapping, ratio_func, weight)
	return result

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
		if isinstance(context, DiGraph) or isinstance(context, MultiDiGraph):
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


def get_default_utility_scalarized_function(
	domain        : AgentType_Domain,
	combiner      : Callable[[dict[AgentType_Name, Utility_Criterion]], float] | None = None,
) -> Utility_Scalarized:
	safe_criteria_dict = {
		k: (
			builder_utility_similarneighbor(
				condition       = lambda x, y: x == y,  #type:ignore
				summand_mapping = lambda x, y: 1.0,
				ratio_func      = ratio_func_basic,
				weight          = None,
			)
			if isinstance(v, list) else
			builder_utility_neighborinrange(
				ratio_func      = ratio_func_basic,
				summand_mapping = lambda x, y: 1.0,
				distance_func   = distance_absolute,
				max_dist        = 1,
				weight          = None,
			)
		)
		for k, v in domain.items()
	}

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
				result       += partial  # TODO sum, average, multiply for different welfares
		return result

	return utility_scalarized

