# tests/real_content: 3 anime slugs have source_url="mp4" instead of file:// path

The 3 anime slugs

- anime_fate_stay_night_fight_15s
- anime_demon_slayer_fight_15s
- anime_jujutsu_kaisen_dialogue_15s

have `"source_url": "mp4"` in the manifest. Likely a Claude Code
mishap during initial manifest population — someone wrote the
extension to the URL field. The intended value is a `file://` path
to a local Blu-ray / Crunchyroll rip.

The bench runner correctly identifies this as `unrecognized source_url scheme: mp4` and skips. Not user-blocking, just a manifest cleanup.

Fix: either populate with actual `file://` paths, or set `source_url=""` until rips are available.
