# Copilot Instructions

This project combines YOLO11 object detection with MiDaS depth estimation for scene perception.

- Keep code small, typed, and testable.
- Prefer pure functions for geometry, association, and schema conversion logic.
- Keep model loading isolated from business logic so tests can run without GPU or model weights.
- Do not commit large model weights, generated images, or local datasets.
- Store reproducible outputs in `artifacts/json/` and visual outputs in `artifacts/images/`.
