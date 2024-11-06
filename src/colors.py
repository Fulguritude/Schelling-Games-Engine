from typing import Literal, get_args

from matplotlib.pyplot import get_cmap   #type:ignore

from src.types import AgentType_Domain, AgentType_ColorMap



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
			result[name] = get_cmap("plasma")
	return result


