from os.path import sep

from src.types import AgentType_NatureProportions

DEFAULT_PADDING            = 0.1    # padding around matplotlib images
DEFAULT_FIGSIZE            = (5, 5) # default matplotlib figure size
DEFAULT_DPI                = 100    # default matplotlib DPI
DEFAULT_GIF_FRAME_DURATION = 200    # default duration of each frame in a gif in ms

MAX_ITERATIONS      =  200  # default maximum number of iterations for the simulation
DEFAULT_HAPPINESS   =  0.5  # default happiness threshold (above => won't move) for agents
DEFAULT_AGENT_NATURES : AgentType_NatureProportions = {  # distribution of agent strategies
	"random"   : 0.,
	"stubborn" : 0.,
	"strategic": 1.,
}

OUT_IMG_DIR = f"out_images" # default output directory for images
TMP_IMG_DIR = f"out_images{sep}tmp_images" # default temporary output directory for images used to compose gifs

DEFAULT_ANTILAG_SLEEP = 0.01 # default sleep time for the attempt at not computing too intensely

SEED = 0
