from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMessageBox, QFileDialog

class AppController(QObject):
    def __init__(self):
        super().__init__()
        
    def setup_ui(self, ui):
        """Setup all UI connections and initial state"""
        self.ui = ui  
        
        # Menu Bar
        ui.actionLoad.triggered.connect(self.on_load_dataset)


        # Connect UI elements to controller methods
        ui.loadDatasetBtn.clicked.connect(self.on_load_dataset)
    def on_load_dataset(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
                self.ui, "Open Dataset", "../data", "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
                )
        if file_path:
            # load data
            pass
