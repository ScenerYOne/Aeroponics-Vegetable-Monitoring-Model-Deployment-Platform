import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  // ดึงค่าจาก .env หรือ Environment Variables ใน Render
  const env = loadEnv(mode, process.cwd(), '');
  
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api/yolo": {
          target: "http://localhost:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/yolo/, ""),
        },
        "/api/keras": {
          target: "http://localhost:8001",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/keras/, ""),
        },
      },
    },
    define: {
      __API_BASE__: JSON.stringify(env.VITE_API_BASE || '/api'),
    },
  }
})