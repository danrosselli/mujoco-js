import { defineConfig } from 'vite';

export default defineConfig({
  optimizeDeps: {
    // onnxruntime-web resolves its .wasm via `new URL(..., import.meta.url)`.
    // Pre-bundling it into .vite/deps breaks that resolution in dev, so let
    // Vite serve the package as-is.
    exclude: ['onnxruntime-web'],
  },
});