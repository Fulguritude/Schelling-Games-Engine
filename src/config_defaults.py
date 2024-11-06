from os.path import sep

from src.types import AgentType_NatureProportions

DEFAULT_PADDING            = 0.1
DEFAULT_FIGSIZE            = (5, 5)
DEFAULT_DPI                = 100
DEFAULT_GIF_FRAME_DURATION = 100

MAX_ITERATIONS      = 200
DEFAULT_HAPPINESS   = 0.5
DEFAULT_BAD_UTILITY = -1.0
DEFAULT_AGENT_NATURES : AgentType_NatureProportions = {
	"random"   : 0.,
	"stubborn" : 0.,
	"strategic": 1.,
}

OUT_IMG_DIR = f"out_images"
TMP_IMG_DIR = f"out_images{sep}tmp_images"

DEFAULT_ANTILAG_SLEEP = 0.3

SEED = 0

