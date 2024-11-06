from __future__ import annotations
from typing import Union, Callable, Literal, Any
from dataclasses import dataclass
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
)
from src.config_defaults import DEFAULT_BAD_UTILITY



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

