'use client'

import { useRouter } from 'next/navigation'
import { useState, useCallback } from 'react'

export default function UploadPage() {
  const router = useRouter()
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleDrop = useCallback(async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    setDragging(false)

    const file = e.dataTransfer.files?.[0]
    if (!file || file.type !== 'application/pdf') {
      alert('PDF 파일만 업로드할 수 있습니다.')
      return
    }

    setLoading(true)
    const formData = new FormData()
    formData.append('file', file)

    const res = await fetch('/api/parse-pdf', {
      method: 'POST',
      body: formData,
    })

    const result = await res.json()
    localStorage.setItem('blocks', JSON.stringify(result.blocks))
    router.push('/editor')
  }, [router])

  return (
    <main className="p-8">
      <h1 className="text-2xl font-bold mb-4">PDF 드래그 앤 드롭</h1>

      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`border-2 border-dashed p-12 rounded text-center transition-all ${
          dragging ? 'bg-blue-50 border-blue-400' : 'bg-white border-gray-300'
        }`}
      >
        {loading
          ? <p className="text-blue-500">업로드 중입니다...</p>
          : <p className="text-gray-500">여기에 PDF 파일을 드래그 앤 드롭하세요</p>
        }
      </div>
    </main>
  )
}
