import { defineConfig } from 'vitest/config';

export default defineConfig({
  // onnxruntime-web/wasm (the CPU-only browser bundle src/web.ts uses) has no Node build; Node
  // runs the web binding on onnxruntime-web's Node build of the same wasm backend instead.
  // scripts/browser-test.mjs covers the browser bundle.
  resolve: { alias: { 'onnxruntime-web/wasm': 'onnxruntime-web' } },
  test: { include: ['test/**/*.test.ts'], testTimeout: 60_000 },
});
