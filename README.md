# scene-perception-yolo11-midas

Prototype scene perception pipeline combining:

- YOLO11 for object detection
- MiDaS for monocular depth estimation
- Association logic that attaches depth statistics to detected objects

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

## Project Layout

- `src/` - pipeline code
- `tests/` - unit tests
- `data/samples/` - local sample inputs
- `artifacts/images/` - generated visual outputs
- `artifacts/json/` - generated structured outputs
