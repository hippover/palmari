from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QSplitter,
    QComboBox,
)
from qtpy.QtCore import QStringListModel
from napari import Viewer
from napari.layers import Image
from palm_tools.processing import RemoveBackgroundFluorescence
import dask.array as da


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.events.connect(self.refresh_possible_inputs)

        input_layout = QVBoxLayout()
        raw_image_dropdown = QComboBox()
        raw_image_dropdown.currentIndexChanged.connect(
            self.input_index_changed
        )
        input_layout.addWidget(QLabel("Input image"))
        input_layout.addWidget(raw_image_dropdown)
        self.input_model = QStringListModel(self.getCandidateLayersNames())
        raw_image_dropdown.setModel(self.input_model)

        remove_bkg_checkbox = QCheckBox("Percentile filtering")
        remove_bkg_percentile = QSpinBox()
        remove_bkg_percentile.valueChanged.connect(self.percentile_changed)
        remove_bkg_percentile.setRange(1, 100)
        remove_bkg_percentile_layout = QHBoxLayout()
        remove_bkg_percentile_layout.addWidget(QLabel("Percentile : "))
        remove_bkg_percentile_layout.addWidget(remove_bkg_percentile)

        remove_bkg_window_size_layout = QHBoxLayout()
        remove_bkg_window_size = QSpinBox()
        remove_bkg_window_size.valueChanged.connect(self.window_size_changed)
        remove_bkg_window_size.setRange(100, 1000)
        remove_bkg_window_size_layout.addWidget(QLabel("Window : "))
        remove_bkg_window_size_layout.addWidget(remove_bkg_window_size)

        remove_bkg_btn = QPushButton("Remove Background")
        remove_bkg_btn.clicked.connect(self._remove_bkg)

        remove_bkg_layout = QVBoxLayout()
        remove_bkg_layout.addWidget(remove_bkg_checkbox)
        remove_bkg_inside_layout = QVBoxLayout()
        remove_bkg_inside_layout.addLayout(remove_bkg_percentile_layout)
        remove_bkg_inside_layout.addLayout(remove_bkg_window_size_layout)
        remove_bkg_inside_layout.addWidget(remove_bkg_btn)
        remove_bkg_inside_container = QWidget()
        remove_bkg_inside_container.setLayout(remove_bkg_inside_layout)
        remove_bkg_layout.addWidget(remove_bkg_inside_container)
        self.remove_bkg_inside_container = remove_bkg_inside_container
        remove_bkg_checkbox.stateChanged.connect(self.enable_bkg_layout)
        remove_bkg_checkbox.setChecked(False)

        main_layout = QVBoxLayout()
        main_layout.addLayout(input_layout)
        main_layout.addWidget(QSplitter())
        main_layout.addLayout(remove_bkg_layout)
        main_layout.addWidget(QSplitter())

        self.setLayout(main_layout)

    def input_index_changed(self, i):
        self.input_image_index = i

    def percentile_changed(self, p):
        self.percentile = p

    def window_size_changed(self, s):
        self.window_size = s

    def enable_bkg_layout(self, a):
        print("Coucou : %s" % a)
        self.remove_bkg_inside_container.setEnabled(a > 0)

    def refresh_possible_inputs(self, event):
        try:
            print("Refresh from %s" % self.__class__)
            print("Event %s" % event)
            self.input_model.setStringList(self.getCandidateLayersNames())
        except Exception as e:
            print(e)

    def getCandidateLayersNames(self):
        return [l.name for l in self.getCandidateLayers()]

    def getCandidateLayers(self):
        all_layers = self.viewer.layers
        image_layers = [
            l
            for l in all_layers
            if type(l) == Image
            and l.source.path is not None
            and "TIF" in l.source.path.upper()
        ]
        return image_layers

    def getInputLayer(self):
        img_layers = self.getCandidateLayers()
        if len(img_layers) == 0:
            return None
        else:
            return img_layers[self.input_image_index]

    def _remove_bkg(self):
        print("remove bkg")
        print(self)
        self.input_layer = self.getInputLayer()
        print("Processing %s" % self.input_layer.name)
        processor = RemoveBackgroundFluorescence(
            percentile=self.percentile, window_size=self.window_size
        )
        print(processor)
        no_bkg = processor.preprocess(
            da.from_array(self.input_layer.data).copy().astype(float)
        )
        self.background_removed_layer = self.viewer.add_image(
            data=no_bkg,
            name=self.input_layer.name + " (background_removed)",
            scale=self.input_layer.scale,
        )

    def _loc(self):
        print("Loc")
