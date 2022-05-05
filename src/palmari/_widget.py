from napari import Viewer
from .processing.tif_pipeline_widget import TifPipelineWidget
from .processing import TifPipeline


class PalmariWidget(TifPipelineWidget):
    def __init__(self, napari_viewer: Viewer):
        tp = TifPipeline.from_dict(
            {
                "name": "napari-default",
                "movie_preprocessors": [{"RemoveBackgroundFluorescence": {}}],
                "loc_processors": [{"DriftCorrector": {}}],
            }
        )
        print(tp)
        super().__init__(tif_pipeline=tp, napari_viewer=napari_viewer)
