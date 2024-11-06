from __future__ import annotations
from os import mkdir
from os.path import exists
import random
from math import log

from src.types import (
	AgentType_Domain,
	AgentType_Vector,
	AgentType_ColorMap,
	AgentType_Constraints,
	Utility_Scalarized,
)
from src.topology import (
	Topology_GridDiagonals,
)
from src.model import (
	SchellingModelConfig_Random,
	SchellingModel,
)
from src.distributions import (
	AgentType_Distributions,
	Distribution_Choice_Categories,
	Distribution_CustomDiscrete,
)
from src.config_defaults import (
	DEFAULT_AGENT_NATURES,
	MAX_ITERATIONS,
	OUT_IMG_DIR,
	TMP_IMG_DIR,
	SEED,
)
from src.utils import set_nice_level
from src.app   import SchellingApp



def example_simple_game() -> SchellingModel:
	grid_topo_dim     = (20, 20)
	agent_amount      = 300
	domain_race_religion_income : AgentType_Domain = {
		"race"     : ["white", "black"],
	}
	colormap_race : AgentType_ColorMap = {
		"race" : {
			"white" : "pink",
			"black" : "brown",
		},
	}
	distribution_race : AgentType_Distributions = {
		"race"     : Distribution_Choice_Categories(["white", "black"]),
	}
	constraints_race_religion_income : AgentType_Constraints | None = None
	utility_race_religion_income     : Utility_Scalarized    | None = None
	model_config = SchellingModelConfig_Random(
		topology      = ("Graph", Topology_GridDiagonals(grid_topo_dim)),
		n_agents      = agent_amount,
		move_mode     = "jump",
		agent_natures = DEFAULT_AGENT_NATURES,
		happiness     = 0.5,
		domain        = domain_race_religion_income,
		constraints   = constraints_race_religion_income,
		utility       = utility_race_religion_income,
		distributions = distribution_race,
		max_iter      = MAX_ITERATIONS,
		colormap      = colormap_race,
		social_net    = None,
	)
	return SchellingModel(model_config)


def example_complex_game() -> SchellingModel:
	grid_topo_dim     = (20, 20)  # (40, 40)
	agent_amount      = 350       # 1500
	domain_race_religion_income : AgentType_Domain = {
		"race"     : ["white", "black", "asian"],
		"religion" : ["christian", "muslim", "jewish"],
		"income"   : (0, 10000000),
	}
	colormap_race_religion : AgentType_ColorMap = {
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
	distributions_race_religion_income : AgentType_Distributions = {
		"race"     : Distribution_Choice_Categories(["white", "black", "asian"]),
		"religion" : Distribution_CustomDiscrete({
			"christian" : 0.6,
			"muslim"    : 0.3,
			"jewish"    : 0.1,
		}),
		#"income" : Distribution_UniformContinuous(0, 10000000),
	}
	constraints_race_religion_income : AgentType_Constraints | None = None

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
		neighbors_with_similar_income = len([
			1 for neighbor_value in neighbor_values
			if abs(log(neighbor_value["income"]) - log(self_value["income"])) <= 0.2  #type:ignore
		])
		income_friendliness_bonus = 0.5 if neighbors_with_similar_income > 0 else 0.0
		result = ratio_same + ratio_similar + income_friendliness_bonus
		return result
			

	model_config = SchellingModelConfig_Random(
		topology      = ("Graph", Topology_GridDiagonals(grid_topo_dim)),
		n_agents      = agent_amount,
		move_mode     = "jump",
		agent_natures = DEFAULT_AGENT_NATURES,
		happiness     = 1.5,
		domain        = domain_race_religion_income,
		constraints   = constraints_race_religion_income,
		utility       = same_race_and_religion_above_all,
		distributions = distributions_race_religion_income,
		max_iter      = MAX_ITERATIONS,
		colormap      = colormap_race_religion,
		social_net    = None,
	)
	return SchellingModel(model_config)




if __name__ == '__main__':
	if not exists(OUT_IMG_DIR):
		mkdir(OUT_IMG_DIR)
	if not exists(TMP_IMG_DIR):
		mkdir(TMP_IMG_DIR)

	set_nice_level(19)
	if SEED is not None:
		random.seed(SEED)
	model1 = example_simple_game()
	model2 = example_complex_game()
	SchellingApp(model2).run()
