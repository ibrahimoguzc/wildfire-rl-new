"""Contains specialized pop-up dialogs for the SoSID Viewer."""

from PyQt5.QtWidgets import QMainWindow, QMessageBox


class StopDialog(QMessageBox):
    """Dialog for confirming if the simulation should be stopped."""

    def __init__(self, parent: QMainWindow):
        super().__init__(parent=parent)
        self.setupUi()

    def setupUi(self) -> None:
        """Performs set-up action to customize the message box."""
        self.setIcon(QMessageBox.Critical)
        self.setWindowTitle("Stop the simulation?")
        self.setText(
            "Do you want to stop the simulation and close the viewer?\n"
            "Once the simulation is stopped, it cannot be started again."
        )
        self.setDetailedText(
            "Pressing 'No' will only close the viewer but keep the "
            "simulation running in the background."
        )
        self.setIcon(QMessageBox.Critical)
        self.setStandardButtons(QMessageBox.No | QMessageBox.Yes)
        self.setEscapeButton(QMessageBox.No)
        self.setDefaultButton(QMessageBox.Yes)
