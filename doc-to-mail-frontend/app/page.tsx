'use client';

import Link from 'next/link';

export default function MainPage() {
  return (
    <main className="flex flex-col items-center justify-center h-screen gap-6 bg-gray-50">
      <h1 className="text-3xl font-bold">공문 자동화 시스템</h1>
      <p className="text-lg text-gray-600">아래에서 원하는 기능을 선택하세요.</p>

      <div className="flex gap-4">
        <Link
          href="/upload"
          className="px-6 py-3 bg-blue-600 text-white rounded-xl shadow-md hover:bg-blue-700 transition"
        >
          PDF 업로드
        </Link>

        <Link
          href="/settings"
          className="px-6 py-3 bg-gray-700 text-white rounded-xl shadow-md hover:bg-gray-800 transition"
        >
          서식 설정
        </Link>
      </div>
    </main>
  );
}