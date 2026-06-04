# v1.0.0 - CMYK GraySpot Detection (CGD)

**Release Date:** June 4, 2026

## Overview

This inaugural release of the HP CMYK Grayspot Detection system delivers a comprehensive machine learning pipeline with advanced training capabilities, optimized dependencies, and a professional graphical interface for inference operations.

---

## Key Features & Enhancements

### 🔬 Training Enhancements
- **Advanced Loss Functions** – Focal Loss implementation for improved class imbalance handling in grayscale spot detection
- **Configurable Training Pipeline** – Phase 2 configuration supporting:
  - Class-weighted loss functions
  - Label smoothing for regularization
  - Stratified K-fold cross-validation for robust model evaluation
- **Hyperparameter Optimization** – Integrated Optuna with Windows threading compatibility
- **Visual Analytics** – Confusion matrix visualization for training analysis

### 🎨 User Interface
- **PyQt6 GUI** – Professional graphical interface for model inference and prediction visualization
- **Flexible Deployment** – Optional GUI dependencies (`pip install -e '.[gui]'`)
- **Intuitive Workflow** – Streamlined inference tabs and model interaction

### 📦 Dependencies & Infrastructure
- **Updated Libraries:**
  - PyTorch Vision ~0.27.0
  - Albumentations ~2.0.8
  - Pandas ~3.0.3
  - UMAP ~0.5.12
  - PyQt6 ~6.11.0
- **Docker Support** – Optimized containerization for streamlined deployment
- **Code Quality** – Automated formatting with Black/Isort

### ✅ Stability & Bug Fixes
- Fixed Optuna smoke test invocation
- Resolved Windows SQLite thread safety concerns
- Enhanced GUI inference tab functionality
- Improved test reliability across platforms

### 📖 Documentation
- Comprehensive README with GUI setup instructions
- Bilingual CI/CD documentation
- Configuration resolution guide
- Test-Driven Development (TDD) strategy documentation

---

## Technical Specifications

| Aspect | Details |
|--------|---------|
| **Language** | Python with HTML-based visualization |
| **License** | MIT |
| **Repository** | [Rackhel/CMYK_MAIN](https://github.com/Rackhel/CMYK_MAIN) |
| **Status** | Production Ready |

---

## Installation

```bash
# Standard installation
pip install -e .

# With GUI support
pip install -e '.[gui]'

# Development environment
pip install -r requirements.txt
```

---

## Getting Started

For detailed setup instructions, training procedures, and configuration options, refer to the [README.md](https://github.com/Rackhel/CMYK_MAIN/blob/main/README.md).

---

## Contributors

Thank you to all team members who contributed to this release.

---

**For issues, questions, or feature requests, please visit the [GitHub Issues page](https://github.com/Rackhel/CMYK_MAIN/issues).**
