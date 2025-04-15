import sys
from PyQt5.QtWidgets import QApplication
from app import AppUI
from app_controller import AppController

def main():
    app = QApplication(sys.argv)
    
    # Create controller first
    controller = AppController()
    
    # Pass controller to UI
    ui_path = "app.ui"  # or get from command line
    window = AppUI(ui_path, controller)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
