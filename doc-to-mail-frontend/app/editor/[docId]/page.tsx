'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const SunEditor = dynamic(() => import('suneditor-react'), { ssr: false });
import 'suneditor/dist/css/suneditor.min.css';

export default function EditorPage() {
  const { docId } = useParams();
  const [blocks, setBlocks] = useState<any[]>([]);
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editedText, setEditedText] = useState<string>('');
  const editorRef = useRef<any>(null);

  const [settingsAll, setSettingsAll] = useState<{ [key: string]: any }>({});
  const [selectedSettingName, setSelectedSettingName] = useState<string>('');
  const [settings, setSettings] = useState<any>(null);

  useEffect(() => {
    const stored = localStorage.getItem('editorSettingsAll');
    const parsed = stored ? JSON.parse(stored) : {};
    setSettingsAll(parsed);
    const firstSetting = Object.keys(parsed)[0] || '';
    setSelectedSettingName(firstSetting);
    setSettings(parsed[firstSetting]);
  }, []);
  function formatTextAsParagraphs(text: string): string {
    return `<p>${text.trim().replace(/<br>+/g, '</p><p>')}</p>`;
  }
  useEffect(() => {
    if (!settings || !docId) return;
    const loadEverything = async () => {
      const res = await fetch(`/api/static/${docId}/meta.json`);
      const meta = await res.json();

      const titleText = meta.find((block: any) => block.type === 'title')?.data?.trim() || '';
      const replaceTitle = (html: string) => html.replace(/{제목}/g, titleText);

      const enriched = await Promise.all(
        meta
          .filter((block: any) => block.type !== 'title')
          .map(async (block: any) => {
            if (block.type === 'text' && block.filename) {
              try {
                const txtRes = await fetch(`/api/static/${docId}/${block.filename}`);
                const text = await txtRes.text();
                const formatted = text
                  .split('\n')
                  .map(line => {
                    const trimmed = line.trim();
                    const leadingSpaces = line.match(/^ */)?.[0].length || 0;
                    const nbspPrefix = '&nbsp;'.repeat(leadingSpaces);
                    const paragraph = `<p>${nbspPrefix}${trimmed}</p>`;

                    if (/^붙임/.test(trimmed)) {
                      return `<p style="line-height:${settings.lineHeight}"><br></p>`+ paragraph
                    }
                    return [paragraph];
                  })
                  .join('');

                return {
                  ...block,
                  text: applyStyleToHTML(formatted, settings)
                };
              } catch {
                return { ...block, text: '' };
              }
            }
            return block;
          })
      );

      const finalBlocks = [
        ...(settings?.beforeBlock
          ? [{ type: 'text', filename: '', text: applyStyleToHTML(formatTextAsParagraphs(replaceTitle(settings.beforeBlock)), settings) }]
          : []),
        ...enriched,
        ...(settings?.afterBlock
          ? [{ type: 'text', filename: '', text: applyStyleToHTML(formatTextAsParagraphs(replaceTitle(settings.afterBlock)), settings) }]
          : [])
      ];

      setBlocks(finalBlocks);
    };

    loadEverything();
  }, [settings, docId]);

  function applyStyleToHTML(text: string, settings: any): string {
    if (!text || !settings) return text;
    const style = `font-family:${settings.fontFamily}; font-size:${settings.fontSize}px; line-height:${settings.lineHeight};`;
    return text.replace(/<p(\s+[^>]*)?>/g, `<p style="${style}">`);
  }

  const insertBlockAt = (index: number, type: 'text' | 'image') => {
    if (type === 'image') {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = 'image/*';
      input.onchange = () => {
        const file = input.files?.[0];
        if (!file) return;
        const url = URL.createObjectURL(file);
        const newBlock = { type: 'image', url };
        setBlocks(prev => [...prev.slice(0, index), newBlock, ...prev.slice(index)]);
      };
      input.click();
      return;
    }
    const newBlock = { type: 'text', filename: '', text: '<p></p>' };
    setBlocks(prev => [...prev.slice(0, index), newBlock, ...prev.slice(index)]);
  };

  const removeBlock = (index: number) => {
    setBlocks(prev => prev.filter((_, i) => i !== index));
    if (editingIndex === index) setEditingIndex(null);
  };

  const saveEditedText = (idx: number) => {
    if (editorRef.current) {
      const newContent = editorRef.current.getContents();
      setBlocks(prev => {
        const updated = [...prev];
        if (updated[idx].type === 'table') {
          updated[idx] = { ...updated[idx], html: newContent };
        } else {
          updated[idx] = { ...updated[idx], text: newContent };
        }
        return updated;
      });
      setEditingIndex(null);
    }
  };

  const copyToClipboard = () => {
    const html = blocks.map(b => {
      if (b.type === 'text') return b.text;
      if (b.type === 'table') return b.html;
      const src = b.url || `/api/static/${docId}/${b.filename}`;
      const width = b.width;
      const height = b.height;
      return `<img src='${src}' style='width: ${width}px; height: ${height === 'auto' ? 'auto' : `${height}px`};' />`;
    }).join('');

    const blob = new Blob([html], { type: 'text/html' });
    const data = [new ClipboardItem({ 'text/html': blob })];
    navigator.clipboard.write(data).then(() => {
      alert('서식이 유지된 HTML 형식으로 복사되었습니다!');
    }).catch(() => {
      alert('복사 실패');
    });
  };

  const handleSettingChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const name = e.target.value;
    setSelectedSettingName(name);
    setSettings(settingsAll[name]);
  };

  return (
    <div className="p-6">
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold">✏️ 메일 내용 편집</h1>
        <div className="space-x-2">
          <select value={selectedSettingName} onChange={handleSettingChange} className="border rounded px-2 py-1 text-sm">
            {Object.keys(settingsAll).map(name => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
          <button onClick={copyToClipboard} className="px-3 py-1 bg-green-500 text-white rounded text-sm hover:bg-green-600">
            📋 전체 복사
          </button>
        </div>
      </div>

      <div className="space-y-6">
        {blocks.map((block, idx) => (
          <div key={idx}>
            <div className="border-t border-gray-300 relative text-center mb-2 space-x-2">
              <button onClick={() => insertBlockAt(idx, 'text')} className="-mt-3 px-2 py-1 bg-white border rounded-full text-sm hover:bg-gray-100">
                ➕ 텍스트 블록 추가
              </button>
              <button onClick={() => insertBlockAt(idx, 'image')} className="-mt-3 px-2 py-1 bg-white border rounded-full text-sm hover:bg-gray-100">
                🖼️ 이미지 블록 추가
              </button>
            </div>
            <div className="border rounded p-4 relative">
              <button onClick={() => removeBlock(idx)} className="absolute top-2 right-2 text-red-500 hover:text-red-700">✖</button>
              {block.type === 'text' || block.type === 'table' ? (
                editingIndex === idx ? (
                  <>
                    <SunEditor
                      key={idx}
                      getSunEditorInstance={(inst) => { editorRef.current = inst; }}
                      setContents={block.text || block.html || ''}
                      onChange={setEditedText}
                      setOptions={{
                        height: '400px',
                        fontSize: [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36],
                        buttonList: [
                          ['undo', 'redo'],
                          ['font', 'fontSize', 'lineHeight'],
                          ['bold', 'underline', 'italic', 'strike'],
                          ['fontColor', 'hiliteColor'],
                          ['align', 'list', 'table'],
                          ['link', 'image'],
                          ['fullScreen', 'codeView']
                        ]
                      }}
                    />
                    <button onClick={() => saveEditedText(idx)} className="mt-2 px-2 py-1 bg-blue-500 text-white text-sm rounded">완료</button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => {
                        setEditingIndex(idx);
                        setEditedText(block.text || block.html || '');
                      }}
                      className="mb-2 px-2 py-1 bg-gray-200 text-sm rounded"
                    >편집</button>
                    {block.type === 'text' ? (
                      <TextBlock key={idx} src={block.filename ? `/api/static/${docId}/${block.filename}` : ''} content={block.text} />
                    ) : (
                      <div className="overflow-auto border my-4" dangerouslySetInnerHTML={{ __html: block.html }} />
                    )}
                  </>
                )
              ) : (
                <img src={block.url || `/api/static/${docId}/${block.filename}`} alt={`Block ${idx}`} className="max-w-full" />
              )}
            </div>
          </div>
        ))}
        <div className="border-t border-gray-300 relative text-center space-x-2">
          <button onClick={() => insertBlockAt(blocks.length, 'text')} className="-mt-3 px-2 py-1 bg-white border rounded-full text-sm hover:bg-gray-100">
            ➕ 텍스트 블록 추가
          </button>
          <button onClick={() => insertBlockAt(blocks.length, 'image')} className="-mt-3 px-2 py-1 bg-white border rounded-full text-sm hover:bg-gray-100">
            🖼️ 이미지 블록 추가
          </button>
        </div>
      </div>
    </div>
  );
}

function TextBlock({ src, content }: { src: string; content?: string }) {
  const [text, setText] = useState(content || '');

  useEffect(() => {
    if (content !== undefined) {
      setText(content);
    } else if (src) {
      fetch(src).then(res => res.text()).then(setText);
    }
  }, [src, content]);

  return (
    <div
      className="sun-editor-editable text-sm font-sans leading-relaxed"
      style={{ whiteSpace: 'normal' }}
      dangerouslySetInnerHTML={{ __html: text }}
    />
  );
}
