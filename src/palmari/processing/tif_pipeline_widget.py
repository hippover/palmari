from __future__ import annotations
from typing import Any, TYPE_CHECKING
import napari
from napari.layers import Image
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QSplitter,
    QGroupBox,
    QFileDialog,
)
import yaml
from functools import partial


from napari.qt import thread_worker
from magicgui.widgets import create_widget, Container, ComboBox, FloatSpinBox
import enum
import dask.array as da
import pandas as pd
import logging

if TYPE_CHECKING:
    from .tif_pipeline import TifPipeline, ProcessingStep


class handled_types(enum.Enum):
    image = "Image"
    points = "Points"
    tracks = "Tracks"


class ProcessingStepWidget(Container):
    def __init__(self, step: ProcessingStep):
        self.widgets_dict = {}
        for param, value in step.__dict__.items():
            logging.debug("Adding widget for %s" % param)
            w_type = (
                step.widget_types[param]
                if param in step.widget_types
                else None
            )
            w_options = (
                step.widget_options[param]
                if param in step.widget_options
                else {}
            )
            widget = create_widget(
                value=value,
                name=param,
                options=w_options,
                widget_type=w_type,
            )
            self.widgets_dict[param] = widget

            widget.changed.connect(partial(step.update_param, param))
        super().__init__(widgets=[w for _, w in self.widgets_dict.items()])


class TifPipelineWidget(QWidget):
    def __init__(
        self, tif_pipeline: TifPipeline, napari_viewer: napari.Viewer
    ):
        super().__init__()

        self.pixel_size = 0.093
        self.delta_t = 0.03
        self.to_export = None
        self.viewer = napari_viewer
        self.tp = tif_pipeline
        self._layers = {}
        self._buttons = {}
        self.setup_ui()

    def activate_buttons(self, last_clicked: int = 0):
        for button_index, button in self._buttons.items():
            enable = button_index <= last_clicked + 1
            button.setEnabled(enable)

    def add_button(self, button: QPushButton):
        index = len(self._buttons)
        self._buttons[index] = button
        return index

    def drop_layers_after(self, first_idx_to_remove: int):
        original_layers = dict(self._layers)
        for idx, existing_layer in original_layers.items():
            if idx >= first_idx_to_remove:
                self.viewer.layers.remove(existing_layer)
                self._layers.pop(idx)
                logging.debug("Removed layer %d" % idx)

    def set_input_image(self, layer: Image = None):

        # Remove subsequent layers
        self.drop_layers_after(1)
        self.to_export = None
        self.export_locs_button.setEnabled(False)

        if layer is not None:
            # Convert data to Dask array
            # Store it
            logging.debug("Set Input Image %s" % layer.name)
            layer.data = layer.data.astype(float)
            if not isinstance(layer.data, da.Array):
                layer.data = da.from_array(layer.data)

            self._layers[0] = layer
            self.rescale_image_layers()

    def rescale_image_layers(self):
        rescaled = False
        for key, layer in self._layers.items():
            if isinstance(layer, Image):
                layer.scale = (1, self.pixel_size, self.pixel_size)
                rescaled = True
        if rescaled:
            self.viewer.reset_view()

    def setup_ui(self):
        main_layout = QVBoxLayout()

        main_layout.addLayout(self.input_layout())

        preprocessing_widget = self.preprocessing_widget()
        if preprocessing_widget is not None:
            main_layout.addWidget(preprocessing_widget)

        main_layout.addWidget(self.localization_widget())

        post_processing_widget = self.locs_processing_widget()
        if post_processing_widget is not None:
            main_layout.addWidget(post_processing_widget)

        main_layout.addWidget(self.tracking_widget())

        main_layout.addStretch(3)

        main_layout.addLayout(self.export_layout())

        self.setLayout(main_layout)

        self.activate_buttons(last_clicked=-1)

    def input_layout(self):
        main_layout = QVBoxLayout()
        input_widget: ComboBox = create_widget(
            annotation=Image,
            name="input_image",
            label="Input image : ",
            options={
                "tooltip": "Raw image on which the pipeline is run",
            },
        )
        input_widget.changed.connect(self.set_input_image)
        self.viewer.layers.events.inserted.connect(input_widget.reset_choices)
        self.viewer.layers.events.changed.connect(input_widget.reset_choices)
        self.viewer.layers.events.removed.connect(input_widget.reset_choices)
        self.viewer.layers.events.reordered.connect(input_widget.reset_choices)
        self.viewer.events.status.connect(input_widget.reset_choices)

        delta_t_widget = FloatSpinBox(
            name="delta_t",
            label="delta t (s)",
            value=self.delta_t,
            step=0.005,
        )

        def set_delta_t(x):
            self.delta_t = x

        delta_t_widget.changed.connect(set_delta_t)

        pixel_size_widget = FloatSpinBox(
            name="pixel_size",
            label="pixel size (um)",
            value=self.pixel_size,
            step=0.001,
        )

        def set_pixel_size(x):
            self.pixel_size = x
            self.rescale_image_layers()

        pixel_size_widget.changed.connect(set_pixel_size)

        container = Container(
            widgets=[input_widget, delta_t_widget, pixel_size_widget]
        )
        main_layout.addWidget(container._widget._qwidget)
        return main_layout

    def export_layout(self):
        main_layout = QVBoxLayout()
        self.export_locs_button = QPushButton(text="Save locs and tracks")
        export_pipeline_button = QPushButton(text="Save pipeline")
        main_layout.addWidget(self.export_locs_button)
        main_layout.addWidget(export_pipeline_button)

        self.export_locs_button.clicked.connect(self.export_locs)
        export_pipeline_button.clicked.connect(self.export_pipeline)

        return main_layout

    def export_locs(self):
        assert self.to_export is not None
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            caption="Export localizations and tracks",
        )
        logging.debug(fileName)
        if fileName is not None and len(fileName) > 2:
            self.to_export.to_csv(fileName, index=True)

    def export_pipeline(self):
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save pipeline parameters",
        )
        logging.debug(fileName)
        if fileName is not None and len(fileName) > 2:
            self.tp.to_yaml(fileName)

    def preprocessing_widget(self):

        if len(self.tp.movie_preprocessors) == 0:
            return None

        main_widget = QGroupBox(title="Image pre-processing")
        main_layout = QVBoxLayout()
        for i, step in enumerate(self.tp.movie_preprocessors):
            step_layout = self.setup_layout_for_step(
                step,
                input_type=handled_types.image,
                output_type=handled_types.image,
            )
            main_layout.addLayout(step_layout)
            if i < len(self.tp.movie_preprocessors) - 1:
                main_layout.addWidget(QSplitter())
        main_widget.setLayout(main_layout)

        return main_widget

    def localization_widget(self):

        main_widget = QGroupBox(
            title="Localization : %s" % self.tp.localizer.name
        )
        main_layout = QVBoxLayout()
        main_layout.addLayout(
            self.setup_layout_for_step(
                self.tp.localizer,
                input_type=handled_types.image,
                output_type=handled_types.points,
                skip_title=True,
            )
        )
        main_widget.setLayout(main_layout)
        return main_widget

    def locs_processing_widget(self):

        if len(self.tp.loc_processors) == 0:
            return None

        main_widget = QGroupBox(title="Post-processing of localizations")
        main_layout = QVBoxLayout()
        for i, step in enumerate(self.tp.loc_processors):
            step_layout = self.setup_layout_for_step(
                step,
                input_type=handled_types.points,
                output_type=handled_types.points,
            )
            main_layout.addLayout(step_layout)
            if i < len(self.tp.loc_processors) - 1:
                main_layout.addWidget(QSplitter())
        main_widget.setLayout(main_layout)

        return main_widget

    def tracking_widget(self):

        main_widget = QGroupBox(title="Tracking : %s" % self.tp.tracker.name)
        main_layout = QVBoxLayout()
        main_layout.addLayout(
            self.setup_layout_for_step(
                self.tp.tracker,
                input_type=handled_types.points,
                output_type=handled_types.tracks,
                skip_title=True,
                is_final_step=True,
            )
        )
        main_widget.setLayout(main_layout)
        return main_widget

    def setup_layout_for_step(
        self,
        step: ProcessingStep,
        input_type: handled_types,
        output_type: handled_types,
        skip_title: bool = False,
        is_final_step: bool = False,
    ) -> QVBoxLayout:
        step_layout = QVBoxLayout()
        if not skip_title:
            step_layout.addWidget(QLabel(step.name))
        step_container = ProcessingStepWidget(step)
        if len(step_container.widgets_dict) > 0:
            step_layout.addWidget(step_container._widget._qwidget)
        step_run_btn = QPushButton(step.action_name)
        btn_index = self.add_button(step_run_btn)
        step_layout.addWidget(step_run_btn)
        input_layer_idx = btn_index
        output_layer_idx = btn_index + 1

        def add_layer(data):
            logging.debug(
                "Adding layer %d -> %s" % (input_layer_idx, output_layer_idx)
            )

            if step.is_localizer:
                print("Scaling positions when adding layer")
                print("Pixel size = %.3f" % self.pixel_size)
                data["t"] = (
                    data["frame"] - data["frame"].min()
                ) * self.delta_t
                data[["x", "y"]] *= self.pixel_size

            self.drop_layers_after(output_layer_idx)
            self.add_layer_of_type(
                data=data,
                data_type=output_type,
                layer_idx=output_layer_idx,
                step_name=step.name,
            )
            self.setEnabled(True)
            self.activate_buttons(last_clicked=btn_index)
            if is_final_step:
                self.to_export = data
            else:
                self.to_export = None
            self.export_locs_button.setEnabled(is_final_step)

        @thread_worker(connect={"returned": add_layer})
        def run_step(checked: bool = True):
            logging.debug(self._layers)
            assert input_layer_idx in self._layers
            self.setEnabled(False)
            if input_type == handled_types.image:
                input_data = self._layers[input_layer_idx].data
            elif input_type == handled_types.points:
                input_data = self._layers[input_layer_idx].result
            elif input_type == handled_types.tracks:
                input_data = self._layers[input_layer_idx].result
            return step.process(input_data)

        step_run_btn.clicked.connect(run_step)

        return step_layout

    def add_layer_of_type(
        self,
        data: Any,
        data_type: handled_types,
        layer_idx: int,
        step_name: str,
    ):
        if data is None:
            logging.debug("Nothing returned by %s" % step_name)

        if data_type == handled_types.image:
            self.add_image_layer(
                returned_image=data, layer_idx=layer_idx, step_name=step_name
            )
        elif data_type == handled_types.tracks:
            self.add_tracks_layer(
                tracks=data, layer_idx=layer_idx, step_name=step_name
            )
        elif data_type == handled_types.points:
            self.add_points_layer(
                points=data, layer_idx=layer_idx, step_name=step_name
            )

    def add_image_layer(self, returned_image, layer_idx: int, step_name: str):

        if layer_idx not in self._layers:
            new_layer = self.viewer.add_image(
                data=returned_image,
                name="%s : %s" % (self._layers[0].name, step_name),
                scale=(1, self.pixel_size, self.pixel_size),
            )
            self._layers[layer_idx] = new_layer
        else:
            self._layers[layer_idx].data = returned_image

    def add_points_layer(self, points, layer_idx: int, step_name: str):

        if layer_idx not in self._layers:
            new_layer = self.viewer.add_points(
                points[["frame", "x", "y"]].values,
                properties=points[
                    ["ratio", "frame", "total_intensity"]
                ].to_dict(),
                symbol="x",
                size=0.25,
                edge_width=0.1,
                face_color="transparent",
                edge_color="ratio",
                edge_contrast_limits=(
                    points.ratio.min(),
                    4 * points.ratio.min(),
                ),
                edge_colormap="rainbow",
                blending="translucent_no_depth",
                name="%s : %s" % (self._layers[0].name, step_name),
            )
            self._layers[layer_idx] = new_layer
        else:
            self._layers[layer_idx].data = points

        self._layers[layer_idx].result = points

    def add_tracks_layer(self, tracks, layer_idx: int, step_name: str):

        if layer_idx not in self._layers:
            new_layer = self.viewer.add_tracks(
                data=tracks[["n", "frame", "x", "y"]],
                tail_width=1,
                tail_length=10,
                head_length=0,
                properties=pd.merge(
                    tracks[["n"]],
                    tracks.groupby("n")
                    .agg({"x": "count", "frame": "min"})
                    .rename(columns={"x": "length", "frame": "start_frame"}),
                    how="left",
                    left_on="n",
                    right_index=True,
                ),
                name="%s : %s" % (self._layers[0].name, step_name),
            )
            self._layers[layer_idx] = new_layer
        else:
            self._layers[layer_idx].tracks = tracks

        self._layers[layer_idx].result = tracks
