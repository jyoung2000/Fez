.PHONY: autoflip-references

# Task 5 — generate naive-baseline reference JSONs for the bench.
# Path B fallback: a center-crop tracker (largest face per second, box
# smoothed) that gives the bench something to compare ClipAI against
# when MediaPipe AutoFlip isn't running. Mark loudly: this is the
# weak comparator; ClipAI being better than naive_baseline is the floor.
autoflip-references:
	docker compose exec backend python -m backend.scripts.run_naive_baseline \
		--manifest tests/real_content/manifest.json \
		--output-dir tests/autoflip_reference_outputs/
