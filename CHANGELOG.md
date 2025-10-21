# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [25.10.0] - 2025-10-21

### Fixed

- **Critical: Incorrect project URLs in setup.py** - Fixed GitHub repository links that were pointing to argon2_cffi instead of fer
- **Integer division bug in Video class** - Changed float division to integer division in classes.py:299 for frame count calculation
- **Duplicate dependency** - Removed duplicate facenet-pytorch entry from setup.py
- **Deprecated Keras API** - Removed deprecated `make_predict_function()` call
- **Deprecated NumPy function** - Changed `np.fromstring()` to `np.frombuffer()` in utils.py
- **Incorrect OpenCV API usage** - Fixed `cv2.rectangle()` call to use correct parameter format (two corner points instead of single 4-tuple)

### Added

- **Enhanced exception classes** - Added `InvalidModelFile` and `FaceDetectionError` exceptions with docstrings
- **Parameter validation** - Added input validation to `FER.__init__()` for scale_factor, min_face_size, min_neighbors, and offsets
- **Better error handling** - Enhanced `load_image()` function with:
  - 10-second timeout for URL downloads
  - More specific error messages
  - Proper type checking with isinstance()
  - Try-except blocks for better error handling
- **Modern packaging** - Added pyproject.toml following PEP 517/518 standards
- **CI/CD automation** - Added GitHub Actions workflow for automated testing across:
  - Multiple OS: Ubuntu, macOS, Windows
  - Python versions: 3.8, 3.9, 3.10, 3.11, 3.12
  - Automated linting with black, ruff, mypy
  - Package building and validation
- **Enhanced .gitignore** - Better patterns for modern tools (ruff, mypy, VS Code, macOS)
- **Development tools** - Enhanced requirements-dev.txt with black, ruff, mypy, and updated versions

### Changed

- **Python version support** - Minimum Python version updated from 3.6 to 3.8 (dropped EOL Python 3.6)
- **Dependency versions** - Added minimum version constraints:
  - matplotlib>=3.5.0
  - opencv-contrib-python>=4.5.0
  - pandas>=1.3.0
  - Pillow>=9.0.0
  - requests>=2.27.0
  - facenet-pytorch>=2.5.0
  - moviepy>=1.0.3
  - ffmpeg-python>=0.2.0 (changed from ffmpeg==1.4)
- **Keras import** - Added fallback import to support both tensorflow.keras and standalone keras
- **Pre-commit configuration** - Updated to modern versions:
  - black 23.3.0 (from 22.1.0)
  - Added ruff for faster linting (replacing flake8)
  - Added mypy for type checking
  - Added standard pre-commit hooks (trailing-whitespace, end-of-file-fixer, etc.)
- **Improved docstrings** - Added comprehensive Google-style docstrings to multiple functions

### Infrastructure

- Added pyproject.toml with full project metadata and tool configurations
- Created GitHub Actions CI/CD pipeline (.github/workflows/ci.yml)
- Enhanced requirements-dev.txt with modern development tools
- Updated .pre-commit-config.yaml with latest tool versions

### Breaking Changes

- **Minimum Python version is now 3.8** (dropped support for Python 3.6 and 3.7)
  - Python 3.6 reached end-of-life in December 2021
  - Users on Python 3.6/3.7 should continue using fer 22.5.1

### Version Format

This release uses Calendar Versioning (CalVer): YY.MM.MICRO
- YY: Short year (25 = 2025)
- MM: Month (10 = October)
- MICRO: Patch number (0 = first release this month)

### Notes

- All public APIs remain unchanged (backward compatible)
- All changes tested with Python 3.10 + TensorFlow + full dependency stack
- Package successfully detects emotions on test images
- Total changes: 8 files modified, 3 files added, +168 lines, -44 lines

---

## [22.5.1] - Previous Release

See git history for previous changes.
