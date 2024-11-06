from __future__ import annotations
from os.path import sep
from time import sleep

from matplotlib.pyplot      import Figure
from kivy.app               import App
from kivy.uix.boxlayout     import BoxLayout
from kivy.uix.slider        import Slider
from kivy.uix.button		import Button
from kivy.uix.togglebutton  import ToggleButton
from kivy.uix.label         import Label
from kivy_garden.matplotlib import FigureCanvasKivyAgg


from .types           import AgentType_Name
from .model           import SchellingModel
from .utils           import combine_img_plots, export_gif_from_pngs
from .config_defaults import (
	TMP_IMG_DIR,
	OUT_IMG_DIR,
	DEFAULT_FIGSIZE,
	DEFAULT_PADDING,
)

class MainWindow(BoxLayout):
	def __init__(self, model: SchellingModel, **kwargs):
		super().__init__(**kwargs)
		self.orientation = 'horizontal'
		self.model = model
		self.model.run()
		self.graph_layout = None
		self.with_labels  = False
		self.with_edges   = False
		# TODO show color legend
		self.build_controls()
		self.render()

	def render(self) -> None:
		if self.graph_layout:
			self.remove_widget(self.graph_layout)
		self.graph_layout = self.create_graph_canvas()
		self.add_widget(self.graph_layout)

	# Export utils

	@staticmethod
	def get_export_filename(iter_step: int, type_name: AgentType_Name) -> str:
		return f"{TMP_IMG_DIR}{sep}schelling_{type_name}_{iter_step}.png"

	def export_png_plot(self, fig: Figure, filename: str) -> None:
		fig.set_size_inches(DEFAULT_FIGSIZE, forward = True)  #type:ignore
		fig.savefig(f"{OUT_IMG_DIR}{sep}{filename}", bbox_inches='tight', pad_inches = DEFAULT_PADDING)

	def export_png_plot_at_iter(self, iter_step: int, type_name: AgentType_Name) -> str:
		fig = self.model.get_figure(
			iter_step   = iter_step,
			type_name   = type_name,
			nodes_pos   = None,
			with_labels = self.with_labels,
			with_edges  = self.with_edges,
		)
		fig_name = MainWindow.get_export_filename(iter_step, type_name)
		self.export_png_plot(fig, fig_name)
		return fig_name

	def export_png_all_plots_at_iter(self, iter_step: int) -> str:
		fig_paths = []
		for type_name in self.model.domain.keys():
			fig_path = self.export_png_plot_at_iter(iter_step, type_name)
			fig_paths.append(fig_path)
		# combine figures into a single plot
		result_fig = combine_img_plots(fig_paths)
		fig_name = f"{TMP_IMG_DIR}{sep}schelling_all_{iter_step}.png"
		result_fig.savefig(fig_name, bbox_inches='tight', pad_inches = DEFAULT_PADDING)
		return fig_name

	def export_gif_plot(self, type_name: AgentType_Name) -> str:
		fig_paths = []
		for iter_step in range(self.model.max_iter):
			print(f"Building {iter_step} image for gif...")
			sleep(0.01)
			fig_path = self.export_png_plot_at_iter(iter_step, type_name)
			fig_paths.append(fig_path)
		gif_name = f"schelling_{type_name}.gif"
		export_gif_from_pngs(fig_paths, gif_name)
		return gif_name

	def export_gif_all_plots(self) -> str:
		fig_paths = []
		for iter_step in range(self.model.max_iter):
			print(f"Building {iter_step} images for gif...")
			sleep(0.01)
			fig_path = self.export_png_all_plots_at_iter(iter_step)
			fig_paths.append(fig_path)
		gif_name = f"schelling_all.gif"
		export_gif_from_pngs(fig_paths, gif_name)
		return gif_name

	# Event Bindings

	def on_export_png(self, instance) -> None:
		iter_step = self.get_iter_step()
		type_name = self.get_selected_type()
		self.export_png_plot_at_iter(iter_step, type_name)

	def on_export_png_all(self, instance) -> None:
		fig_paths = []
		for type_name in self.model.domain.keys():
			fig = self.model.get_figure(
				iter_step   = self.get_iter_step(),
				type_name   = type_name,
				nodes_pos   = None,
				with_labels = self.with_labels,
				with_edges  = self.with_edges,
			)
			fig_name = f"{TMP_IMG_DIR}{sep}schelling_{type_name}.png"
			fig.savefig(fig_name, bbox_inches='tight', pad_inches = DEFAULT_PADDING)
			fig_paths.append(fig_name)
		# combine figures into a single plot
		result_fig = combine_img_plots(fig_paths)
		result_fig.savefig(f"schelling_all.png", bbox_inches='tight', pad_inches = DEFAULT_PADDING)

	def on_export_gif(self, instance) -> None:
		type_name = self.get_selected_type()
		self.export_gif_plot(type_name)

	def on_export_gif_all(self, instance) -> None:
		self.export_gif_all_plots()



	def on_iter_value_change(self, instance, value):
		self.slider_ilabel.text = f"Iteration step: {int(value)}"
		self.iter_value = int(value)
		self.render()

	def on_toggle_labels(self, instance):
		self.with_labels        = not self.with_labels
		self.toggle_labels.text = "Show labels" if not self.with_labels else "Hide labels"
		self.model.figures      = {}  # TODO better way to force rerender through cache invalidation
		self.render()

	def on_toggle_edges(self, instance):
		self.with_edges        = not self.with_edges
		self.toggle_edges.text = "Show edges"   if not self.with_edges  else "Hide edges"
		self.model.figures     = {}  # TODO better way to force rerender through cache invalidation
		self.render()

	def on_type_radio_selected(self, instance):
		instance.state = "down"
		self.render()

	# Dashboard builders

	@staticmethod
	def build_radiobutton(
		labels     : Iterable[str],
		group_name : str,
		binding    : Callable[[ToggleButton], None],
	) -> BoxLayout:
		radiobutton_layout = BoxLayout(orientation='horizontal')
		buttons = [
			ToggleButton(
				text  = label,
				group = group_name,
				state = ("down" if i == 0 else "normal"),
			)
			for i, label in enumerate(labels)
		]
		for button in buttons:
			button.bind(on_press = binding)
			radiobutton_layout.add_widget(button)
		return radiobutton_layout


	def build_export_buttons(self) -> BoxLayout:
		export_layout  = BoxLayout(orientation='horizontal')
		export_png     = Button(text="Export PNG")
		export_png_all = Button(text="Export PNG all")
		export_gif     = Button(text="Export GIF")
		export_gif_all = Button(text="Export GIF all")
		export_png     .bind(on_press = self.on_export_png)
		export_png_all .bind(on_press = self.on_export_png_all)
		export_gif     .bind(on_press = self.on_export_gif)
		export_gif_all .bind(on_press = self.on_export_gif_all)
		export_layout.add_widget(export_png)
		export_layout.add_widget(export_png_all)
		export_layout.add_widget(export_gif)
		export_layout.add_widget(export_gif_all)
		return export_layout


	def build_toggles(self) -> BoxLayout:
		toggles_layout     = BoxLayout(orientation='horizontal')
		self.toggle_edges  = ToggleButton(text = "Show edges"  if not self.with_edges  else "Hide edges"  )
		self.toggle_labels = ToggleButton(text = "Show labels" if not self.with_labels else "Hide labels" )
		self.toggle_edges  .bind(on_press = self.on_toggle_edges)
		self.toggle_labels .bind(on_press = self.on_toggle_labels)
		toggles_layout.add_widget(self.toggle_edges)
		toggles_layout.add_widget(self.toggle_labels)
		return toggles_layout

	def build_iterator_slider(self) -> BoxLayout:
		iter_slider_layout = BoxLayout(orientation='vertical')
		self.slider_ilabel = Label(text="Iteration step: 0")
		self.slider_iters  = Slider(min = 0, max = self.model.max_iter, value = 0)
		self.slider_iters  .bind(value = self.on_iter_value_change)
		iter_slider_layout.add_widget(self.slider_iters)
		iter_slider_layout.add_widget(self.slider_ilabel)
		return iter_slider_layout

	def build_controls(self) -> None:
		controls_layout    = BoxLayout(orientation='vertical')
		self.iter_slider   = self.build_iterator_slider()
		self.type_radio	   = MainWindow.build_radiobutton(self.model.domain.keys(), "type_radio", self.on_type_radio_selected)
		self.toggles       = self.build_toggles()
		self.exports       = self.build_export_buttons()
		controls_layout.add_widget(self.iter_slider)
		controls_layout.add_widget(self.type_radio)
		controls_layout.add_widget(self.toggles)
		controls_layout.add_widget(self.exports)
		self.add_widget(controls_layout)


	def create_graph_canvas(self) -> FigureCanvasKivyAgg:
		fig = self.model.get_figure(
			iter_step   = self.get_iter_step(),
			type_name   = self.get_selected_type(),
			nodes_pos   = None,
			with_labels = self.with_labels,
			with_edges  = self.with_edges,
		)
		result           = FigureCanvasKivyAgg(fig)
		return result

	def get_iter_step(self) -> int:
		result = int(self.slider_iters.value)
		return result

	def get_selected_type(self) -> AgentType_Name:
		result = [button.text for button in self.type_radio.children if button.state == "down"][0]
		return result


class SchellingApp(App):
	def __init__(self, model: SchellingModel, **kwargs):
		super().__init__(**kwargs)
		self.model = model

	def build(self):
		return MainWindow(self.model)
