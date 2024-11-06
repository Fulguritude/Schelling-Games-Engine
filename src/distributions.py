from __future__ import annotations
from typing import Union, Callable
from dataclasses import dataclass
import random

from src.types import (
	AgentType_Name,
	AgentType_Value_Discrete,
	AgentType_Value_Continuous,
)



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
