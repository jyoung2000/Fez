# bench: panel_joebudden_10s fails jitter only — multi-speaker tuning ticket

`panel_joebudden_10s` scored within spec on every axis except jitter
(0.109 vs spec ≤ 0.05). All other axes clean.

Single-axis fail on multi-speaker / multi-cut content suggests a
specific solver knob: likely Kalman process noise, or solver
λ_acc/λ_jerk weights for the multi_speaker_panel content type in
`reframe_config.py`.

Validate any fix on `panel_breakfast_club_10s` first — that clip
currently passes with jitter=0.025 (right at spec). A jitter-reducing
change must not push breakfast_club above spec.

Blocks: graduating multi_speaker_panel for default-on without per-clip
caveats. Does not block Phase C wiring or Phase C allowlist for the
opt-in flag.
