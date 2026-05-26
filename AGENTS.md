# Project summary

This is a Computer Vision MVP for urban scene perception.

The project combines:
- YOLO11 object detection
- MiDaS monocular depth estimation
- per-object relative depth scoring

The system processes a single RGB image and outputs:
- annotated detection image
- depth heatmap
- combined visualization
- JSON file with detections and relative depth scores

# Core rule

Depth is relative, not metric. Never describe MiDaS output as real distance in meters.

Use terms:
- relative depth
- approximate relative depth score

Do not use:
- distance in meters
- real-world distance
- calibrated depth

# Architecture

- src/schemas.py: shared dataclasses and output schema
- src/detection.py: YOLO11 wrapper
- src/depth.py: MiDaS wrapper
- src/association.py: bbox-to-depth association
- src/visualization.py: rendering and side-by-side output
- src/cli.py: command-line entrypoint
- tests/: unit tests for pure logic

# Commands

Install:
pip install -r requirements.txt

Run:
python -m src.cli --image data/samples/street.jpg --output artifacts

Test:
pytest -q

Lint:
ruff check .

Format:
ruff format .

# Coding standards

- Prefer small functions.
- Prefer typed dataclasses for detection results.
- Keep model wrappers separate from pure data processing logic.
- Do not train models.
- Do not add Streamlit, Docker, video, or batch mode during MVP.
- Do not introduce large datasets into Git.
- Do not commit model weights manually.
- Use median depth inside bbox, not mean, unless explicitly changed.

# Validation before commit

Before committing:
1. Run ruff check .
2. Run pytest -q
3. Run one smoke test on a sample image
4. Check that artifacts are generated correctly