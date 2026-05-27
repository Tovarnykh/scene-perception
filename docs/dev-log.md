# Development log

## 2026-05-10 — Project bootstrap

### Goal
Build MVP for single-image urban scene perception using YOLO11 and MiDaS.

### MVP scope
- Input: one RGB image
- Output: detection overlay, depth heatmap, combined image, JSON result
- No training
- No video
- No Streamlit
- No Docker in MVP

### Key technical decision
Use relative depth score only. No metric distance in meters.

### Next step
Implement schemas and bbox-depth association first, because those can be tested without models.