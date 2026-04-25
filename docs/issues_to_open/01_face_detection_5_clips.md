# bench: face detection produces empty dense_faces on 5 of 9 real clips

The `measure_human_reframe_quality` real-content path scored 9 clips on
2026-04-23. 5 produced the all-zeros pattern indicating an empty
dense_faces list reached the per-frame scoring loop:

- panel_verzuz_tank_tyrese_10s
- sports_nba_fastbreak_20s
- sports_f1_onboard_20s
- music_chris_brown_performance_15s
- music_beyonce_solo_narrative_15s

Stderr signals:
- `[DenseFaces] FFmpeg extraction failed (rc=183)`
- `mediapipe/gpu/gl_context_egl.cc:303 RET_CHECK failure ... eglMakeCurrent() returned error 0x3008`
- `[mov,mp4,m4a,3gp,3g2,mj2 @ 0x...] moov atom not found` (Beyoncé clip is a known dead URL — separate issue)

Hypotheses:
1. Codec mismatch — trim used `-c copy` so source codec preserved; YouTube serves a mix of H.264, VP9, AV1.
2. MediaPipe GPU context init fails inside the container; behavior may diverge per-clip on the CPU fallback path.
3. moov atom corruption from the trim's stream-copy.

Diagnostic plan: see `docs/pr_path_a_face_detection.md` for the prompt that drives the fix.

Blocks: full bench coverage. Does not block Phase C wiring.
