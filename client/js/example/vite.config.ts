import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

export default defineConfig({
  // relative URLs, so the build works under any path (GitHub Pages: /livekit-wakeword/)
  base: './',
  // the package from this checkout; an app would `npm install @livekit/wakeword onnxruntime-web`
  resolve: {
    alias: { '@livekit/wakeword': fileURLToPath(new URL('../src/web.ts', import.meta.url)) },
  },
  // serves models/hey_livekit.onnx from the repo root next to the page
  // the package's own node_modules (onnxruntime-web) are outside this folder
  server: { fs: { allow: ['..'] } },
  publicDir: fileURLToPath(new URL('../../../models', import.meta.url)),
  // onnxruntime-web finds its .wasm files next to its own module; pre-bundling would move it
  optimizeDeps: { exclude: ['onnxruntime-web'] },
});
