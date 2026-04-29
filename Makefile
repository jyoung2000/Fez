.PHONY: autoflip-references autoflip-references-build naive-references

# Task 5A — generate REAL MediaPipe AutoFlip references via the
# Docker sidecar. First build takes ~30-60 minutes; subsequent builds
# reuse the build cache. See infra/autoflip/README.md.
autoflip-references-build:
	docker compose --profile bench build autoflip

autoflip-references: autoflip-references-build
	@docker image inspect clipai/autoflip:local >/dev/null 2>&1 \
		|| (echo "AutoFlip image not built. Run 'make autoflip-references-build' first."; exit 1)
	python -m backend.scripts.run_autoflip_reference \
		--manifest tests/real_content/manifest.json \
		--output-dir tests/autoflip_reference_outputs/

# Path B fallback — naive center-crop baseline. Used when MediaPipe
# AutoFlip can't be built in the homelab. The bench rollup labels
# these references "naive_baseline" so SOTA claims aren't muddled.
naive-references:
	docker compose exec backend python -m backend.scripts.run_naive_baseline \
		--manifest tests/real_content/manifest.json \
		--output-dir tests/autoflip_reference_outputs/
