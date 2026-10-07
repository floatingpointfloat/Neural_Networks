import sys

from gui import create_app

if __name__ == "__main__":
    app, window = create_app()
    sys.exit(app.exec())
