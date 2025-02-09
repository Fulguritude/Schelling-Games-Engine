from matplotlib.pyplot import Figure, subplots, imread
from PIL import Image
import os
import platform

from src.config_defaults import DEFAULT_GIF_FRAME_DURATION




def combine_img_plots(fig_paths: list[str]) -> Figure:
	result_fig, axes = subplots(1, len(fig_paths))
	for ax, fig_path in zip(axes, fig_paths):
		img = imread(fig_path)
		ax.imshow(img)
		ax.axis('off')
	return result_fig


def export_gif_from_pngs(
	png_paths    : list[str],
	gif_filename : str,
	duration     : int | None = None
) -> None:
	duration = duration if duration is not None else DEFAULT_GIF_FRAME_DURATION
	images = [Image.open(png) for png in png_paths]
	images[0].save(
		gif_filename, 
		save_all      = True, 
		append_images = images[1:], 
		duration      = duration,
	)


def set_nice_level(level):
	if platform.system() in ["Linux", "Darwin"]:
		try:
			os.nice(level)
			print(f"Set process niceness level to {level}")
		except PermissionError as e:
			print("Permission denied: Unable to set niceness level. Try running with appropriate privileges.")
			raise e
	else:
		print(f"os.nice() is not supported on this operating system ({platform.system()}).")
