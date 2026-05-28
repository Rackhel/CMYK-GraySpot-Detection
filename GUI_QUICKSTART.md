# CMYK PyQt6 Desktop GUI Quick Start

## Local Setup

```bash
pip install -r requirements.txt
python main.py
```

Alternative entry point:

```bash
python -m gui.main
```

## PyInstaller

```bash
pip install pyinstaller
python build_gui_executable.py
dist/grayspot/grayspot.exe  # Windows
dist/grayspot/grayspot      # Linux
```

## GUI Structure

```text
gui/
├── main.py
├── main_window.py
├── workers/
├── services/
├── tabs/
├── components/
├── dialogs/
├── styles/
├── resources/
└── utils/
```

## Architecture Notes

- PyQt6 desktop application only.
- `MainWindow` owns tab orchestration and high-level signal routing.
- Tabs collect input and display results.
- Services create worker boundaries.
- Workers run long operations on QThread and communicate only through Qt signals.
- Backend functionality in `src/` stays black-boxed from the GUI.
