# `backend/models/`

Binary model files that ship with the ClipAI backend. These are
committed to the repo rather than downloaded at build time so the
Docker image is self-contained and deterministic.

## Files

### `lbpcascade_animeface.xml`

- **Size:** ~250 KB
- **Source:** <https://github.com/nagadomi/lbpcascade_animeface>
- **License:** Public domain (per Nagadomi's stated terms on the
  upstream repo).
- **Used by:** `backend/services/anime_face_detector.py` via the
  `DEFAULT_CASCADE_PATH` module constant. Loaded lazily by
  `detect_anime_faces()` on the first anime-classified clip.
- **Why it's needed:** YuNet / FaceMesh / OpenCV DNN all struggle on
  stylized anime faces because they're trained on human-face
  distributions. The lbpcascade was trained specifically on anime /
  cartoon character data and catches the stylized eyes + simplified
  nose geometry that the human detectors drop. Week 2 wiring
  (`CLIPAI_ANIME_FACE_DETECTOR=1`) augments the live-action dense
  face stream with these detections on frames where the live-action
  detector found nothing or only low-confidence boxes.

If you find this file and wonder why it's here, it's because the
anime face detector no-ops without it — `detect_anime_faces()`
returns `AnimeDetectionResult(skipped_reason="cascade not found: ...")`
when the XML is missing. Do **not** delete it.

### `light_asd.onnx` (Phase C)

- **Auto-downloaded:** lazily by `backend/services/light_asd.py` from
  the upstream Light-ASD release on first use when
  `CLIPAI_ASD_BACKEND=light_asd` is set. Pulled to
  `backend/models/light_asd.onnx`.
- **Used by:** the v3 active speaker timeline
  (`build_active_speaker_timeline_v3`) for audio-visual ASD that
  handles overlapping speech and off-camera speakers — both v2-impossible.
- **CPU-only** via onnxruntime CPUExecutionProvider. Default OFF; the
  v2 lip-aperture heuristic stays as the production path until
  Jalon's content-corpus validation confirms parity or improvement.

### `yolo11n.pt` and `yolo11n-pose.pt` (Phase B)

- **Auto-downloaded:** by `ultralytics` on first use when
  `CLIPAI_OBJECT_DETECTOR=yolo11n` and/or
  `CLIPAI_PERSON_USE_POSE=true` are set. We don't commit the weights;
  the lazy loaders fetch them to `backend/models/` (or `/data/models/`)
  on demand.
- **Used by:** `backend/services/object_detector.py` (yolo11n.pt) and
  `backend/services/person_detector.py` (yolo11n-pose.pt for the
  head-anchor extraction).
- **Why:** YOLO11n has ~22% fewer params than YOLOv8m at better small
  -object mAP; the pose variant gives 17 COCO keypoints per person so
  back-turned / profile / far-subject frames get a real head anchor
  instead of a torso-biased bbox center.
- **CPU-only.** Both flags default OFF.

### `insightface/models/buffalo_s/` (ArcFace 512-d)

- **Size:** ~16 MB (recognition pack only; full buffalo_s is ~25 MB)
- **Source:** Auto-fetched by `insightface.app.FaceAnalysis(name='buffalo_s')`
  on first use. The Dockerfile pre-downloads it so cold containers don't
  stall on the first analysis run.
- **Used by:** `backend/services/face_detector.py`
  `_extract_face_embeddings_arcface()`.
- **Why:** SFace 128-d under-separates identities — ArcFace 512-d gives
  substantially tighter clusters and reduces speaker-slot flips on
  multi-speaker fixtures. Enabled via the `CLIPAI_FACE_EMBEDDING=arcface`
  env var (default `sface`). CPU-only via the onnxruntime
  `CPUExecutionProvider` — must NOT touch the GPU (Whisper / Ollama
  share the GTX 1650).

## How to re-download

```bash
curl -L -o backend/models/lbpcascade_animeface.xml \
  https://github.com/nagadomi/lbpcascade_animeface/raw/master/lbpcascade_animeface.xml
# Verify
head -c 32 backend/models/lbpcascade_animeface.xml  # should start with <?xml
wc -l backend/models/lbpcascade_animeface.xml       # should be ~6693 lines
```
