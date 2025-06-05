# Copyright (c) 2018. Deutsches Zentrum fuer Luft- und Raumfahrt (DLR).
# All rights reserved.  http://www.dlr.de/sl

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/

"""Contains the main SoSID GUI classes and functions."""

from __future__ import annotations

import datetime
import inspect
import math
import os
import sys
import time
from collections.abc import Callable, Generator, Sequence
from functools import cached_property
from pathlib import Path

import qdarkstyle
import qtawesome as qta
from deprecated.sphinx import deprecated
from IPython import get_ipython
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QWidget
from pyqtgraph import GraphicsLayout, setConfigOptions
from pyqtgraph.graphicsItems import ViewBox

from sosid.abstract import Viewable, Viewer
from sosid.gui.dialogs import StopDialog
from sosid.gui.exporters import ImageExporter, SVGExporter
from sosid.gui.items import AxisItem, BorderItem
from sosid.paths import ICONS_PATH, TEMPLATES_PATH
from sosid.simulation import Simulation

EXPORTER_MAP = {
    "SVG": SVGExporter,
    "PNG": ImageExporter,
}
"""Map of export format to the responsible Exporter."""

uiFile = TEMPLATES_PATH / "viewer.ui"
WindowTemplate, TemplateBaseClass = uic.loadUiType(uiFile)

# Setting the Locale settings to United Kindom
QtCore.QLocale.setDefault(
    QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedKingdom)
)

# TODO look up how to implement multithreading
# TODO add intersphinx for Qt

__all__ = ["Viewer2D", "display"]


class SimulationItem:
    """Initializes the GUI representation protocol into a Qt object."""

    __slots__ = ["item", "object", "on_update", "painter"]

    def __init__(self, object, painter, on_init, on_update, z_order, scale):
        self.object = object
        self.painter = painter
        self.on_update = on_update
        self.item = on_init(painter, object)
        self.item.setScale(scale)
        self.item.setZValue(z_order)

    def update(self):
        """Updates the Qt object using :py:attr:`on_update`."""
        self.on_update(self.item, self.object)

    @property
    def isUpdatable(self) -> bool:
        """Defines if :py:class:`SimulationItem` requires update."""
        return self.on_update is not None


ElementUpdateMethod = Callable[[QMainWindow], None]
"""Signature used for updating a :py:class:WindowElement`."""


class WindowElement(cached_property):
    """Specifies a cached element of :py:class:`QMainWindow`.

    The intended usage is to decorate a method that instantiates a
    :py:class:`Qt.QWidgets.QWidget` and adds itself to the correct
    position on the :py:class:`Qt.QWidgets.QMainWindow::

        @WindowElement
        def fpsLabel(self) -> QtWidgets.QLabel:
            '''Label used to display frames-per-second.'''
            label = QtWidgets.QLabel("FPS: ")
            self.statusbar.addPermanentWidget(label)
            return label

    Attrs:
        updateMethod: A callable unbound method assigned to update
            the :py:class:`Qt.QWidgets.QWidget` that the
            :py:class:`WindowElement` wraps.
    """

    __slots__ = ()  # Removes the instance dictionary to increase performance

    updateMethod: ElementUpdateMethod | None = None

    def onUpdate(self, method: ElementUpdateMethod) -> ElementUpdateMethod:
        """Registers ``method`` as the updator of the element.

        This is intended to be used as a decorator after the
        :py:class:`WindowElement` is specified. With the following
        usage, the decorated method will be automatically called when
        the :py:class:`Qt.QWidgets.QMainWindow` is updated::

            @fpsLabel.on_update
            def _fpsLabelUpdator(self) -> None:
                fps = 1 / ((now := time.time()) - self.last_update)
                self.last_update = now
                self.fpsLabel.setText(f"FPS: {int(fps)}")
        """
        self.updateMethod = method
        return method

    def update(self, window: QMainWindow) -> None:
        """Updates the :py:class:`WindowElement` on ``window``."""
        self.updateMethod(window) if self.isUpdatable else ()

    @property
    def isUpdatable(self) -> bool:
        """Defines if :py:class:`WindowElement` requires updating."""
        return self.updateMethod is not None

    def __get__(self, instance, owner=None):
        """Injects :py:class:`WindowElement` into :py:class:`Qt.QWidget`.

        The control is the :py:class:`Qt.QWidget` that the
        py:class:`WindowElement` wrapped method returns. This
        injection is done to preserve the mapping between the
        :py:class:`Qt.QWidget` and the :py:class:`WindowElement.
        """
        if (obj := super().__get__(instance, owner)) is not self:
            if not hasattr(obj, "windowElement"):
                obj.windowElement = self
        return obj


class UIWindow(WindowTemplate):
    """Defines the :py:class:`QMainWindow` containing all UI elements.

    Note:
        These additional UI elements cannot be created easily with the
        Qt Designer and it is useful to have them explicitely defined
        for IDE code-completion functionality. Currently, the order
        in which they are defined is the order they take in
        the :py:class:`Qt.QWidgets.QWidget`.
    """

    updatableWindowElements: list[WindowElement] = []

    def __init__(self, viewer: Viewer2D):
        self.viewer = viewer
        super().__init__()
        self.setupUi(parent=viewer)

        if not self.simulation.parameters.reset_available:
            del self.resetButton

    @property
    def simulation(self) -> Simulation:
        """Provides easy access to the :py:class:`Simulation`."""
        return self.viewer.simulation

    @cached_property
    def simulationControls(self) -> tuple[QWidget, ...]:
        return (
            self.startButton,
            self.pauseButton,
            self.resumeButton,
            self.stopButton,
            self.stepControl,
            self.resetButton,
            self.timestepControl,
            self.slaveCheckBox,
            self.zoomControl,
        )

    @WindowElement
    def graphicsLayout(self) -> GraphicsLayout:
        """Adds the :py:class:`GraphicsLayout` required by PyQtGraph."""
        return GraphicsLayout()

    @WindowElement
    def viewBox(self) -> ViewBox:
        """Adds a `viewBox` attribute into a ``ui`` object.

        This is necessary as adding GraphicsLayout and ViewBox objects
        in the Qt Designer are not possible.
        """
        viewBox = self.graphicsLayout.addViewBox(
            # Inverting Y-Axis is done to preserve Qt coordinate-system
            lockAspect=True,
            invertY=True,  # CHANGE WITH CAUTION!
        )
        self.graphicsView.setCentralItem(viewBox)
        return viewBox

    @WindowElement
    def border(self) -> BorderItem:
        ax, ay = self.simulation.environment.origin
        width, height = self.simulation.environment.active_area
        border = BorderItem(ax=ax, ay=ay, width=width, height=height)
        border.setZValue(math.inf)
        self.viewBox.addItem(border)
        return border

    @WindowElement
    def xaxis(self) -> AxisItem:
        """X-axis indicator arrow and label."""
        ax = AxisItem(label="x", angle=0, color="r")
        ax.setZValue(math.inf)
        self.viewBox.addItem(ax)
        return ax

    @WindowElement
    def yaxis(self) -> AxisItem:
        """Y-axis indicator arrow and label."""
        ax = AxisItem(label="y", angle=90, color="g")  # CW+
        ax.setZValue(math.inf)
        self.viewBox.addItem(ax)
        return ax

    @WindowElement
    def fpsLabel(self) -> QtWidgets.QLabel:
        """Label used to display frames-per-second."""
        label = QtWidgets.QLabel("FPS: ")
        self.statusbar.addPermanentWidget(label)
        return label

    last_update = time.time()

    @fpsLabel.onUpdate
    def updateFpsLabel(self) -> None:
        fps = 1 / ((now := time.time()) - self.last_update)
        self.last_update = now
        self.fpsLabel.setText(f"FPS: {int(fps)}")

    @WindowElement
    def runtimeLabel(self) -> QtWidgets.QLabel:
        """Label used to display :py:class:`Simulation` runtime."""
        label = QtWidgets.QLabel("Simulation Runtime: ")
        self.statusbar.addPermanentWidget(label)
        return label

    @runtimeLabel.onUpdate
    def updateRuntimeLabel(self) -> None:
        self.runtimeLabel.setText(
            f"Simulation Runtime: {self.simulation.timer.runtime}"
        )

    @WindowElement
    def missiontimeLabel(self) -> QtWidgets.QLabel:
        """Label used to display :py:class:`Simulation` mission time."""
        label = QtWidgets.QLabel("Current Time: ")
        self.statusbar.addPermanentWidget(label)
        return label

    @missiontimeLabel.onUpdate
    def updateMissiontimeLabel(self) -> None:
        mission_time = self.simulation.timer.mission_time.strftime(
            "%d %B %Y %H:%M:%S"
        )
        self.missiontimeLabel.setText(f"Current  Mission Time: {mission_time}")

    @WindowElement
    def startButton(self) -> QtWidgets.QPushButton:
        """Button that starts the :py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Start")
        self.statusbar.addWidget(button)
        self.inactive_options = [
            {
                "color": "light blue",
                "opacity": 1.0,
            }
        ]
        button.clicked.connect(self.enable_buttons)
        button.clicked.connect(self.simulation.start)
        button.clicked.connect(self.disable_start_button)
        button.setIcon(
            QtGui.QIcon(
                qta.icon(
                    "mdi.play-circle-outline", options=self.inactive_options
                )
            )
        )
        return button

    @WindowElement
    def resumeButton(self) -> QtWidgets.QPushButton:
        """Button that resumes the py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Resume")
        self.statusbar.addWidget(button)
        button.clicked.connect(self.simulation.play)
        self.active_options = [
            {
                "color": "light blue",
                "color_active": "red",
                "opacity": 1.0,
            }
        ]
        button.setIcon(
            QtGui.QIcon(
                qta.icon("mdi.step-forward", options=self.active_options)
            )
        )
        button.setEnabled(False)
        return button

    @WindowElement
    def pauseButton(self) -> QtWidgets.QPushButton:
        """Button that pauses the :py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Pause")
        self.statusbar.addWidget(button)
        button.clicked.connect(self.simulation.pause)
        button.setIcon(
            QtGui.QIcon(
                qta.icon(
                    "mdi.pause-circle-outline", options=self.active_options
                )
            )
        )
        button.setEnabled(False)
        return button

    @WindowElement
    def stopButton(self) -> QtWidgets.QPushButton:
        """Button that stops the :py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Stop")
        self.statusbar.addWidget(button)
        button.clicked.connect(self.simulation.stop)
        button.setIcon(
            QtGui.QIcon(
                qta.icon(
                    "mdi.stop-circle-outline", options=self.active_options
                )
            )
        )
        button.setEnabled(False)
        return button

    @WindowElement
    def resetButton(self) -> QtWidgets.QPushButton:
        """Button that resets the :py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Reset")
        self.statusbar.addWidget(button)
        button.clicked.connect(self.simulation.reset)
        button.clicked.connect(self.enable_start_button)
        button.clicked.connect(self.disable_buttons)

        button.setIcon(
            QtGui.QIcon(qta.icon("mdi.restart", options=self.inactive_options))
        )
        button.setEnabled(False)
        return button

    @WindowElement
    def timestepControl(self) -> QtWidgets.QDoubleSpinBox:
        """Control that modifies the :py:attr:`time_step`."""
        box = QtWidgets.QDoubleSpinBox()
        box.setValue(self.simulation.time_step.total_seconds())
        box.setRange(0.01, 2.00)
        box.setPrefix("Time step: ")
        box.setSuffix(" [s]")
        box.setSingleStep(0.01)
        self.statusbar.addWidget(box)
        box.valueChanged.connect(self.on_timestep_change)
        return box

    @WindowElement
    def slaveCheckBox(self) -> QtWidgets.QCheckBox:
        """Checkbox that slaves :py:meth:`step` method to the GUI."""
        box = QtWidgets.QCheckBox("Slaving ")
        box.setChecked(slave := self.simulation.parameters.slaving)
        # Set simulation to slave based on user input
        self.on_slave_changed(slave)
        self.statusbar.addWidget(box)
        box.stateChanged.connect(self.on_slave_changed)
        return box

    @WindowElement
    def stepControl(self) -> QtWidgets.QSpinBox:
        """Control that sets the number of steps in slaved operation."""
        box = QtWidgets.QSpinBox()
        box.setValue(10)
        box.setRange(1, 200)
        box.setPrefix("Steps per frame: ")
        self.statusbar.addWidget(box)
        return box

    @WindowElement
    def exportButton(self) -> QtWidgets.QPushButton:
        """Button that snapshots the :py:class:`Simulation`."""
        button = QtWidgets.QPushButton("Export")
        self.statusbar.addWidget(button)
        button.clicked.connect(self.export)
        button.setIcon(
            QtGui.QIcon(qta.icon("mdi.export", options=self.active_options))
        )
        return button

    @WindowElement
    def exportOptions(self) -> QtWidgets.QComboBox:
        """Determines export format of the :py:class:`Simulation`."""
        comboBox = QtWidgets.QComboBox()
        for file_ext in EXPORTER_MAP:
            comboBox.addItem(file_ext)
        self.statusbar.addWidget(comboBox)
        return comboBox

    @WindowElement
    def zoomControl(self) -> QtWidgets.QSpinBox:
        """Control that modifies the zoom level of icons"""
        box = QtWidgets.QSpinBox()
        box.setRange(1, 1000)
        box.setValue(100)
        box.setPrefix("Zoom: ")
        box.setSuffix(" %")
        box.setSingleStep(10)
        self.statusbar.addWidget(box)
        box.valueChanged.connect(self.on_zoom_change)
        return box

    @cached_property
    def simulation_path(self) -> Path:
        """Filepath of the simulation class."""
        return Path(inspect.getfile(self.simulation.__class__)).parent

    @cached_property
    def figure_path(self) -> Path:
        """Filepath of exported figures."""
        path = self.simulation_path / "figures"
        path.mkdir(parents=True, exist_ok=True)  # Ensure path exists
        return path

    def enable_start_button(self):
        self.startButton.setEnabled(True)

    def disable_start_button(self):
        self.startButton.clearFocus()
        self.startButton.setDisabled(True)

    def enable_buttons(self) -> None:
        """Enables Resume, Pause and Stop buttons."""
        self.resumeButton.setEnabled(True)
        self.pauseButton.setEnabled(True)
        self.stopButton.setEnabled(True)
        self.resetButton.setEnabled(True)

    def disable_buttons(self) -> None:
        """Enables Resume, Pause and Stop buttons."""
        self.resetButton.clearFocus()
        self.resumeButton.setDisabled(True)
        self.pauseButton.setDisabled(True)
        self.stopButton.setDisabled(True)
        self.resetButton.setDisabled(True)

    def export(self) -> None:
        """Export the current view to an image file.

        Filepath is relative to the directory the py:class:`Simulation`
        is defined in.
        """
        file_ext = self.exportOptions.currentText()
        file_name = (
            f"{self.simulation.name}-"
            f"{datetime.datetime.now().strftime('%m.%d.%Y.%H.%M.%S')}"
            f".{file_ext.lower()}"
        )
        exporter = EXPORTER_MAP[file_ext](self.viewBox.scene())
        file_set = self.figure_path / file_name
        # ImageExporter uses string whereas the SVG exporter uses
        # directory itself
        if EXPORTER_MAP[file_ext] == ImageExporter:
            file_set = str(file_set)
        exporter.export(file_set)

    def on_timestep_change(self) -> None:
        """Updates the :py:attr:`Simulation.time_step`."""
        self.simulation.time_step = datetime.timedelta(
            seconds=self.timestepControl.value()
        )

    def on_zoom_change(self) -> None:
        """Updates the :py:attr:`Simulation.scale_factor`."""
        self.simulation.parameters.zoom_scale_factor = (
            self.zoomControl.value() / 100.0
        )

    def on_slave_changed(self, state) -> None:
        """Sets the state of the :py:attr:`Simulation.not_slaved`."""
        if state == QtCore.Qt.Checked or state is True:
            self.simulation.not_slaved.clear()
        else:
            self.simulation.not_slaved.set()

    def setupUi(self, parent: QMainWindow) -> None:
        """Initializes :py:class:`WindowElement` and performs setup."""
        super().setupUi(parent)
        self.graphicsView.setBackground(None)
        for name, element in self.getWindowElementsItems():
            getattr(self, name)  # Triggers evaluation of the WindowElement
            if element.isUpdatable:
                self.updatableWindowElements.append(element)

    def update(self) -> None:
        """Updates all :py:class:`WindowElement` that require it."""
        for element in self.updatableWindowElements:
            element.update(window=self)

    def deleteSimulationControls(self) -> None:
        """Deletes all simulation :py:class:`Qt.QWidget` controls.

        Note:
            If a simulation control has a registered
            :py:meth:`WindowElement.onUpdate` method then it will also
            be removed from :py:attr:`updatableWindowElements`.
        """
        updatables = self.updatableWindowElements
        for control in self.simulationControls:
            try:
                control.deleteLater()
            except RuntimeError:
                # Handling control already being deleted
                pass
            if control.windowElement in updatables:
                updatables.pop(updatables.index(control.windowElement))

    @classmethod
    def getWindowElementsItems(
        cls,
    ) -> Generator[tuple[str, WindowElement], None, None]:
        """Gets :py:class:`WindowElement` attribute names/elements."""
        for name, item in vars(cls).items():
            if isinstance(item, WindowElement):
                yield name, item


class ResolvedMeta(type(TemplateBaseClass), type(Viewer)):
    """Creates a resolved metaclass for the Template and Viewer.

    If this resolved metaclass is not created, then :py:class:`Viewer2D`
    cannot be instantiated due to a metaclass conflict.
    """


class Viewer2D(TemplateBaseClass, Viewer, metaclass=ResolvedMeta):
    """Defines a 2D Qt :py:class:`Viewer` to visualize simulations."""

    itemMap: dict[Viewable, SimulationItem] = {}
    updateableItemMap: dict[Viewable, SimulationItem] = {}

    def __init__(
        self, simulation: Simulation, parent=None, max_fps: int = 200, **kwargs
    ):
        super().__init__(parent=parent)
        setConfigOptions(**kwargs)
        self.max_fps = max_fps
        self.simulation = simulation
        # Embeds the Viewer2D instance in the Simulation object
        simulation.__view__ = self
        self.ui = UIWindow(viewer=self)
        self.compileSimulationItems()
        self.renderSimulationItems()
        self.configureUI()
        self.update()

    # TODO allow user to specify configuration options
    def configureUI(self):
        """Applies specified configuration options."""
        self.setWindowTitle(f"SoSID Viewer: {self.simulation.name}")
        self.setWindowIcon(QtGui.QIcon(os.path.join(ICONS_PATH, "logo.svg")))
        self.showMaximized()

    def compileSimulationItems(self) -> None:
        """Compiles map between viewables and their simulation items."""
        viewables = {}
        for obj in vars(self.simulation).values():
            if isinstance(obj, Viewable):
                gui_repr = obj.__gui_repr__()
                if isinstance(gui_repr, Sequence):
                    viewables.update(
                        {g["object"]: SimulationItem(**g) for g in gui_repr}
                    )
                else:
                    viewables[obj] = SimulationItem(**gui_repr)
        self.itemMap = viewables
        self.updateableItemMap = self.getUpdatableItems()

    def renderSimulationItems(self) -> None:
        """Renders :py:class:`SimulationItem` in :py:attr:`items`."""
        for i in self.itemMap.values():
            self.ui.viewBox.addItem(i.item)

    def addSimulationItem(self, viewable: Viewable):
        """Allows runtime addition of a :py:class:`Viewable`."""
        gui_repr = viewable.__gui_repr__()
        self.itemMap[viewable] = (simItem := SimulationItem(**gui_repr))
        self.ui.viewBox.addItem(simItem.item)  # Actually displaying the item
        if simItem.isUpdatable:
            self.updateableItemMap[viewable] = simItem

    def removeSimulationItem(self, viewable: Viewable):
        """Removes the provided ``viewable``, a :py:class:`Viewable`."""
        simItem = self.itemMap.pop(viewable, None)
        self.updateableItemMap.pop(viewable, None)
        if simItem:
            self.ui.viewBox.removeItem(simItem.item)

    def getUpdatableItems(self) -> dict[Viewable, SimulationItem]:
        """Gets all :py:class:`Viewable` with :py:attr:`on_update`."""
        return {
            viewable: item
            for viewable, item in self.itemMap.items()
            if item.isUpdatable
        }

    @cached_property
    def frame_delay(self) -> int:
        """Duration to wait for the next frame in SI millisecond."""
        return 1000 // self.max_fps if self.max_fps else 1

    def update(self):
        """Updates all :py:class:`SimulationItem` and the UI."""
        for simItem in self.updateableItemMap.values():
            simItem.update()
        self.ui.update()

        # Runs the simulation for a set number of steps when slaved
        if (
            not self.simulation.not_slaved.is_set()
            and not self.simulation.is_stopped.is_set()
        ):
            self.simulation.step_for(n_steps=self.ui.stepControl.value())
        QtCore.QTimer.singleShot(self.frame_delay, self.update)

    def closeEvent(self, event) -> None:
        """Asks the user if the simulation should be stopped.

        If the user selects "Yes" then the :py:class:`Simulation` is
        stopped and all controls are deleted.

        Note:
            This dialog does not display if the simulation was already
            stopped then the dialog will not be displayed.
        """
        if not self.simulation.is_stopped.is_set():
            if not self.simulation.is_alive():
                event.ignore()
            elif (dlg := StopDialog(parent=self)).exec() == dlg.Yes:
                self.simulation.stop()
                self.ui.deleteSimulationControls()
        event.accept()


_sosid_gui_instances = {}


def display(simulation: Simulation):
    """Launches the SoSID Qt-based GUI for a ``simulation`` instance.

    Args:
        simulation: A SoSID :py:class:`Simulation` instance.

    Returns:
        A tuple containing the launched Qt objects.

        - [0] :py:class:`QtWidgets.QApplication` instance
        - [1] :py:class:`Viewer2D` window instance

    """
    # Creating a Qt application instance if one does not already exist
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Viewer2D(simulation, imageAxisOrder="row-major")

    # Adding compatibility for IPython event-loop integration
    # Thanks to: https://stackoverflow.com/questions/54952165/
    if ipython := get_ipython():
        ipython.magic("gui qt5")
        window.show()
    else:
        window.show()
        app.exec()  # Adds support for VS Code PTSVD Debugger

    # Storing Qt application and window to prevent garbage collection
    _sosid_gui_instances[simulation] = {"app": app, "window": window}

    return app, window


def get_instances() -> dict[Simulation, dict[str, object]] | None:
    """Gets all active instances of the SoSID Qt-based GUI.

    This function returns the module-level variable
    `_sosid_gui_instances`, a dictionary containing the QApplication and
    the window. The dictionary can be accessed by providing specific
    :py:class:`Simulation` instances.

    Returns:
        Dictionary containing all GUI instances if any instance exist,
        otherwise :py:data:`None`.

    """
    return _sosid_gui_instances if _sosid_gui_instances else None


@deprecated(version="0.15.0", reason="Use the new `display()` function")
def main():
    sys.path.insert(0, os.getcwd())
    from examples.wildfire.simulation import (
        WildfireParameters,
        WildfireSimulation,
    )

    sim = WildfireSimulation(parameters=WildfireParameters(), seed=0)

    # Adding compatibility for IPython event-loop integration
    # Thanks to: https://stackoverflow.com/questions/54952165/
    ipython = get_ipython()
    if ipython:
        ipython.magic("gui qt5")

    # Creating a Qt application instance if one does not already exist
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    window = Viewer2D(sim, imageAxisOrder="row-major")
    window.show()
    return app, window
