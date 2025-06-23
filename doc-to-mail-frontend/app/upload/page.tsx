'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function UploadPage() {
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    if (file && file.type !== 'application/pdf') {
      setError('PDF 파일만 업로드할 수 있습니다.');
      return;
    }
    setSelectedFile(file);
    setError(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('업로드 실패');

      const result = await response.json();
      console.log('처리 결과:', result);
      router.push(`/editor/${result.docId}`);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-2xl font-bold mb-6">📄 공문 업로드</h1>

      <label className="mb-4 cursor-pointer px-4 py-2 bg-gray-100 border border-gray-300 rounded hover:bg-gray-200 text-sm">
        {selectedFile ? selectedFile.name : '📎 PDF 파일 선택'}
        <input
          type="file"
          accept="application/pdf"
          onChange={handleFileChange}
          className="hidden"
        />
      </label>

      {error && <p className="text-red-500 mb-2">{error}</p>}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || isUploading}
        className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
      >
        {isUploading ? '업로드 중...' : '변환 시작'}
      </button>
    </div>
  );
}
