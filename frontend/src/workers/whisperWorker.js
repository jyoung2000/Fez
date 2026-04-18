import { pipeline } from '@huggingface/transformers';

let transcriber = null;
let currentModel = null;

self.onmessage = async ({ data }) => {
  const { type, audio, model, language } = data;

  if (type === 'load') {
    try {
      self.postMessage({ type: 'status', status: 'loading', message: `Loading ${model}...` });

      // Detect WebGPU availability inside worker
      let device = 'wasm';
      let dtype = 'q8';
      try {
        if (navigator.gpu) {
          const adapter = await navigator.gpu.requestAdapter();
          if (adapter) {
            device = 'webgpu';
            dtype = { encoder_model: 'fp32', decoder_model_merged: 'q4' };
          }
        }
      } catch (gpuErr) {
        console.warn('WebGPU detection failed in worker, falling back to WASM:', gpuErr);
      }

      transcriber = await pipeline('automatic-speech-recognition', model, {
        device,
        dtype,
        progress_callback: (progress) => {
          if (progress.status === 'progress') {
            self.postMessage({
              type: 'status', status: 'downloading',
              message: `Downloading model: ${Math.round(progress.progress || 0)}%`,
              progress: progress.progress,
            });
          }
        },
      });
      currentModel = model;
      self.postMessage({ type: 'status', status: 'ready', device });
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  }

  if (type === 'transcribe') {
    if (!transcriber) {
      self.postMessage({ type: 'error', error: 'Model not loaded' });
      return;
    }
    try {
      self.postMessage({ type: 'status', status: 'transcribing', message: 'Transcribing...' });
      const startTime = performance.now();

      const result = await transcriber(audio, {
        return_timestamps: 'word',
        chunk_length_s: 30,
        stride_length_s: 5,
        // Only pass language if explicitly set; undefined = auto-detect
        ...(language ? { language } : {}),
      });

      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      self.postMessage({
        type: 'result',
        result: {
          text: result.text,
          chunks: result.chunks || [],
          language: language || result.language || 'auto',
        },
        elapsed,
      });
    } catch (err) {
      self.postMessage({ type: 'error', error: err.message });
    }
  }

  if (type === 'dispose') {
    if (transcriber) {
      await transcriber.dispose();
      transcriber = null;
      currentModel = null;
    }
    self.postMessage({ type: 'status', status: 'disposed' });
  }
};
