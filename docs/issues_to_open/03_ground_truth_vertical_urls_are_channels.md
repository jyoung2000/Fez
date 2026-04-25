# tests/real_content: ground_truth_vertical_url is a channel handle for 5 slugs

The following slugs have channel handles in their `ground_truth_vertical_url`,
which yt-dlp can't resolve to a single matching vertical clip:

- panel_joebudden_10s    →  https://www.youtube.com/@JoeBuddenClips
- panel_breakfast_club_10s →  https://www.tiktok.com/@breakfastclubam
- sports_nba_fastbreak_20s →  https://www.tiktok.com/@nba
- sports_boxing_8s        →  https://www.tiktok.com/@daznboxing
- music_chris_brown_performance_15s →  https://www.tiktok.com/@chrisbrownofficial

Each needs a single specific watch?v= / /video/ URL pointing at the
vertical cut of the same moment as the source clip. Without these the
human-parity bench can't run (it requires per-clip ORB-homography
trajectories from a matching human edit).

Two paths:
1. Manual curation — visit each channel, find the matching vertical, paste URL into manifest.
2. Build `scripts/find_ground_truth_pairs.py` that takes a horizontal source URL + a vertical channel handle and uses yt-dlp's `--dump-json --playlist-end 200` plus title-keyword + publish-date + duration-ratio scoring to suggest top 5 candidates.

Blocks: human-parity bench (mae_cx, framing_macro_f1, cut_timing_deviation, lead_room_correlation metrics). Does not block the quality bench.
