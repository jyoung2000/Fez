.PHONY: autoflip-references autoflip-references-build naive-references download-light-asd-model

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

# Task C — download the Light-ASD ONNX model + print its sha256.
# Paste the printed hash into Dockerfile / Dockerfile.gpu's
# LIGHT_ASD_SHA256 env var to pin the supply-chain artifact.
# Idempotent: re-running just re-prints the existing file's hash
# unless --force is passed.
download-light-asd-model:
	@mkdir -p backend/models
	@if [ ! -f backend/models/light_asd.onnx ]; then \
		echo "Downloading Light-ASD ONNX model..."; \
		curl --retry 4 --retry-delay 5 --retry-all-errors -fsSL \
			-o backend/models/light_asd.onnx \
			"https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd.onnx"; \
	else \
		echo "backend/models/light_asd.onnx already present (use 'rm' to refetch)."; \
	fi
	@echo "── Pin this hash in LIGHT_ASD_SHA256 (Dockerfile + Dockerfile.gpu): ──"
	@sha256sum backend/models/light_asd.onnx
