import sys
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from core.models.base import ModelFactory

def main():
    """Main application entry point"""
    # Initialize translation models
    ModelFactory.get_model("translation", source_lang="en", target_lang="fr")
    ModelFactory.get_model("translation", source_lang="en", target_lang="ar")

    # Create and run application
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
