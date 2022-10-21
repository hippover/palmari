from __future__ import annotations
from qtpy.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
)
from qtpy import QtCore

from magicgui.widgets import LineEdit

from .tif_pipeline import TifPipeline


class PipelineEditor(QDialog):
    def __init__(self, tif_pipeline: TifPipeline):
        super().__init__()

        self.initial_tp_dict = dict(tif_pipeline.to_dict())
        self.tp = tif_pipeline
        self.setup_ui()
        self.selected_class(None)
        self.setModal(True)

    def setup_ui(self):
        main_layout = QVBoxLayout()

        self.down_icon = self.style().standardIcon(
            getattr(QStyle, "SP_ArrowDown")
        )
        self.up_icon = self.style().standardIcon(getattr(QStyle, "SP_ArrowUp"))
        self.in_icon = self.style().standardIcon(
            getattr(QStyle, "SP_DialogYesButton")
        )
        self.out_icon = self.style().standardIcon(
            getattr(QStyle, "SP_DialogNoButton")
        )

        def update_tp_name(s):
            self.tp.name = s

        self.pipeline_name_field = LineEdit()
        self.pipeline_name_field.value = self.tp.name
        self.pipeline_name_field.changed.connect(update_tp_name)
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name :"))
        name_layout.addWidget(self.pipeline_name_field._widget._qwidget)
        main_layout.addLayout(name_layout)

        # self.steps_view = LayersSelectorWidget(self.tp, self)
        self.steps_view = QTreeWidget()
        self.steps_view.setHeaderLabel("Steps")

        self.upButton = QPushButton(self.up_icon, "")
        self.downButton = QPushButton(self.down_icon, "")
        self.addButton = QPushButton("Add")
        self.removeButton = QPushButton("Remove")

        self.upButton.clicked.connect(self.move_step_up)
        self.downButton.clicked.connect(self.move_step_down)
        self.addButton.clicked.connect(self.add_step)
        self.removeButton.clicked.connect(self.remove_step)

        doneButton = QPushButton("Done")
        doneButton.setDefault(True)
        doneButton.clicked.connect(self.accept)

        cancelButton = QPushButton("Cancel")
        cancelButton.clicked.connect(self.cancel)

        main_layout.addWidget(self.steps_view)
        self.setLayout(main_layout)

        self.populate_steps_view()

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addWidget(self.upButton)
        action_buttons_layout.addWidget(self.downButton)
        action_buttons_layout.addSpacing(3)
        action_buttons_layout.addWidget(self.addButton)
        action_buttons_layout.addWidget(self.removeButton)
        action_buttons_layout.addSpacing(3)
        action_buttons_layout.addWidget(cancelButton)
        action_buttons_layout.addWidget(doneButton)

        main_layout.addLayout(action_buttons_layout)

    def populate_steps_view(self):
        try:
            self.steps_view.currentItemChanged.disconnect()
        except TypeError as e:
            pass

        self.steps_view.clear()
        for step_type, steps in self.tp.available_steps.items():
            step_type_item = QTreeWidgetItem()
            # Indicate that it cannot be selected, using Flags
            step_type_item.setFlags(QtCore.Qt.ItemIsEnabled)
            step_type_item.setText(0, step_type)
            for (step_name, step_class) in steps:
                step_item = QTreeWidgetItem()
                step_item.setText(0, step_name)
                step_item.setData(0, 0, step_name)
                step_item.setData(0, 1, step_class)
                step_item.setIcon(
                    0,
                    self.in_icon
                    if self.tp.contains_class(step_name)
                    else self.out_icon,
                )
                step_type_item.addChild(step_item)
            self.steps_view.addTopLevelItem(step_type_item)
        self.steps_view.currentItemChanged.connect(self.selected_class)
        self.steps_view.itemSelectionChanged.connect(self.selected_index)
        self.steps_view.expandAll()
        self.selected_index()

    def cancel(self):
        self.tp = TifPipeline.from_dict(self.initial_tp_dict)
        self.reject()

    def selected_index(self):
        if len(self.steps_view.selectedItems()) == 0:
            self.addButton.setDisabled(True)
            self.removeButton.setDisabled(True)
            self.upButton.setDisabled(True)
            self.downButton.setDisabled(True)

    def selected_class(self, step_item: QTreeWidgetItem):
        try:
            step = step_item.text(0)
        except AttributeError:
            return
        already_contained = self.tp.contains_class(step)
        if already_contained:
            print("TP contains %s" % step)
        else:
            print("TP does not contain %s" % step)
        can_be_removed = self.tp.can_be_removed(step)
        is_mandatory = self.tp.is_mandatory(step)
        there_are_several_options = self.tp.has_alternatives_to(step)
        self.addButton.setEnabled(not already_contained)
        self.addButton.setText("Add" if not is_mandatory else "Use instead")
        self.removeButton.setEnabled(already_contained and can_be_removed)
        self.upButton.setEnabled(
            already_contained
            and not is_mandatory
            and there_are_several_options
        )
        self.downButton.setEnabled(
            already_contained
            and not is_mandatory
            and there_are_several_options
        )

    def selected_step_name(self) -> str:
        try:
            step_item = self.steps_view.selectedItems()[0]
            step_name = step_item.text(0)
            return step_name
        except IndexError as e:
            return None

    def move_step_up(self):
        step_name = self.selected_step_name()
        if step_name is None:
            return
        step_type = self.tp.step_type_of(step_name)
        old_index = self.tp.index_of(step_name)
        new_index = max(old_index - 1, 0)
        self.tp[step_type].insert(new_index, self.tp[step_type].pop(old_index))
        self.populate_steps_view()

    def move_step_down(self):
        step_name = self.selected_step_name()
        if step_name is None:
            return
        step_type = self.tp.step_type_of(step_name)
        old_index = self.tp.index_of(step_name)
        new_index = max(old_index + 1, len(self.tp[step_type]) - 1)
        self.tp[step_type].insert(new_index, self.tp[step_type].pop(old_index))
        self.populate_steps_view()

    def add_step(self):
        step_name = self.selected_step_name()
        if step_name is None:
            return
        step_type = self.tp.step_type_of(step_name)
        # print("Trying to add %s" % step_name)
        step_class = self.tp.step_class_of(step_name)
        step = step_class()
        # print("Got %s" % step)
        # print(step_type)
        if step_type == "movie_preprocessors":
            self.tp.movie_preprocessors.append(step)
        elif step_type == "loc_processors":
            self.tp.loc_processors.append(step)
        elif step_type == "detector":
            self.tp.detector = step
        elif step_type == "localizer":
            self.tp.localizer = step
        elif step_type == "tracker":
            self.tp.tracker = step

        print(self.tp.to_dict())

        self.populate_steps_view()

    def remove_step(self):
        step_name = self.selected_step_name()
        if step_name is None:
            return
        assert self.tp.can_be_removed(step=step_name)
        step_type = self.tp.step_type_of(step_name)
        if step_type == "movie_preprocessors":
            self.tp.movie_preprocessors = self.remove_step_from_list(
                self.tp.movie_preprocessors, step_name
            )
        elif step_type == "loc_processors":
            self.tp.loc_processors = self.remove_step_from_list(
                self.tp.loc_processors, step_name
            )
        self.populate_steps_view()

    def remove_step_from_list(self, l, step):
        names_l = [str(s.__class__).split(".")[-1][:-2] for s in l]
        return [s for s, name in zip(l, names_l) if step != name]
