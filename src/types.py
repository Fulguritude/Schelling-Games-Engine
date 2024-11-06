from typing import (
	Any,
	Union,
	Callable,
	Literal,
	TypedDict,
	Hashable,
)

from networkx import (
	Graph,
	DiGraph,
	MultiGraph,
	MultiDiGraph,
)
from matplotlib.colors import Colormap

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
	Graph,
	DiGraph,
	MultiGraph,
	MultiDiGraph,
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

