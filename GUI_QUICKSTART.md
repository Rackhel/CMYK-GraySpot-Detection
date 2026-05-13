# Quick Start Guide for GUI Development

## Installation

### 1. Local Setup (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Run GUI locally
streamlit run gui/app.py

# Open browser to http://localhost:8501
```

### 2. Docker Setup (Reproducible)

```bash
# Build image
docker build -t grayspot:latest .

# Run GUI
docker run --rm -it -p 8501:8501 \
  -v ${PWD}/data_set:/app/data_set \
  -v ${PWD}/outputs:/app/outputs \
  grayspot:latest \
  streamlit run gui/app.py

# Open browser to http://localhost:8501
```

### 3. PyInstaller (Executable for Reviewers)

```bash
# Install PyInstaller
pip install pyinstaller

# Build executable
python build_gui_executable.py

# Run
dist/grayspot/grayspot.exe  # Windows
dist/grayspot/grayspot      # Linux
```

---

## GUI Structure

```
gui/
├── app.py                 # Main Streamlit entry
├── pages/                 # Multi-page tabs
│   ├── 01_Dashboard.py    # Model metrics
│   ├── 02_Inference.py    # Image upload & predict
│   ├── 03_Model_Info.py   # Training history
│   └── 04_Configuration.py # Settings
├── components/            # Reusable UI components
│   ├── metrics_display.py
│   ├── chart_builder.py
│   └── image_viewer.py
└── assets/                # Config & styling
    └── config.json
```

---

## Development Notes

- **Streamlit**: Fast iteration, reload on file save
- **Pages**: Auto-discovered from `pages/` folder (01_name.py format)
- **Components**: Modular, reusable widgets
- **TODO markers**: Search codebase for `TODO` to see what to implement

---

## Docker vs PyInstaller

| Use Case | Method |
|----------|--------|
| **Develop locally** | `streamlit run gui/app.py` |
| **Test reproducibility** | Docker: `docker run ...` |
| **Ship to reviewers** | PyInstaller `.exe` |
| **Server deployment** | Docker + optional Compose |

---

## Team Customization

The GUI is a **skeleton** — your team can:
- Modify page layouts
- Add/remove tabs
- Change color scheme
- Integrate real model inference
- Add custom business logic

All marked with **TODO** comments.
