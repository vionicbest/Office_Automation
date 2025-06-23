import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*", // 프론트에서 이 경로로 요청
        destination: "http://backend:8000/:path*", // 실제 백엔드 주소
      },
    ];
  },
};

export default nextConfig;
