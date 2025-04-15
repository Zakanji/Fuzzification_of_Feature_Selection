import sys
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QApplication, QMainWindow

class AppUI(QMainWindow):
    def __init__(self, ui_file, controller=None):
        super().__init__()
        loadUi(ui_file, self)
        self.controller = controller
        if self.controller:
            self.controller.setup_ui(self)

if __name__ == "__main__":
    ui_path = "app.ui"
    if len(sys.argv) > 1:
        ui_path = sys.argv[1]
    app = QApplication(sys.argv)
    window = AppUI(ui_path)
    window.show()
    sys.exit(app.exec_())
