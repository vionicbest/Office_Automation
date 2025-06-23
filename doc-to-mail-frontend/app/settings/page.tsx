'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';

const SunEditor = dynamic(() => import('suneditor-react'), { ssr: false });
import 'suneditor/dist/css/suneditor.min.css';

export default function EditorSettingsPage() {
  const [settingName, setSettingName] = useState('');
  const [selectedSettingName, setSelectedSettingName] = useState('');
  const [allSettings, setAllSettings] = useState<Record<string, any>>({});

  const [fontFamily, setFontFamily] = useState('Arial');
  const [fontSize, setFontSize] = useState(14);
  const [lineHeight, setLineHeight] = useState(1.75);
  const [beforeBlock, setBeforeBlock] = useState('<p>수신: ...</p>');
  const [afterBlock, setAfterBlock] = useState('<p>끝.</p>');

  const editorBeforeRef = useRef<any>(null);
  const editorAfterRef = useRef<any>(null);

  // 설정 불러오기
  useEffect(() => {
    const saved = localStorage.getItem('editorSettingsAll');
    if (saved) {
      const parsed = JSON.parse(saved);
      setAllSettings(parsed);
    }
  }, []);

  const handleSelectChange = (name: string) => {
    setSelectedSettingName(name);
    const setting = allSettings[name];
    if (setting) {
      setFontFamily(setting.fontFamily || 'Arial');
      setFontSize(setting.fontSize || 14);
      setLineHeight(setting.lineHeight || 1.75);
      setBeforeBlock(setting.beforeBlock || '<p>수신: ...</p>');
      setAfterBlock(setting.afterBlock || '<p>끝.</p>');
    }
  };

  const saveNewSetting = () => {
    if (!settingName.trim()) {
      alert('설정 이름을 입력하세요.');
      return;
    }
    const updated = {
      ...allSettings,
      [settingName]: {
        fontFamily, fontSize, lineHeight, beforeBlock, afterBlock
      }
    };
    setAllSettings(updated);
    localStorage.setItem('editorSettingsAll', JSON.stringify(updated));
    alert(`"${settingName}" 설정이 저장되었습니다.`);
    setSettingName('');
  };

  const updateCurrentSetting = () => {
    if (!selectedSettingName) {
      alert('수정할 설정을 선택하세요.');
      return;
    }
    const updated = {
      ...allSettings,
      [selectedSettingName]: {
        fontFamily, fontSize, lineHeight, beforeBlock, afterBlock
      }
    };
    setAllSettings(updated);
    localStorage.setItem('editorSettingsAll', JSON.stringify(updated));
    alert(`"${selectedSettingName}" 설정이 수정되었습니다.`);
  };

  const deleteCurrentSetting = () => {
    if (!selectedSettingName) return;
    const updated = { ...allSettings };
    delete updated[selectedSettingName];
    setAllSettings(updated);
    localStorage.setItem('editorSettingsAll', JSON.stringify(updated));
    alert(`"${selectedSettingName}" 설정이 삭제되었습니다.`);
    setSelectedSettingName('');
  };

  return (
    <div className="p-6 max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">🛠️ 전체 공문 구조 및 서식 설정</h1>

      <div className="flex items-center gap-4">
        <label className="font-semibold">📂 저장된 설정 불러오기:</label>
        <select
          value={selectedSettingName}
          onChange={(e) => handleSelectChange(e.target.value)}
          className="border p-2 rounded"
        >
          <option value="">선택하세요</option>
          {Object.keys(allSettings).map(name => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
        <button
          onClick={deleteCurrentSetting}
          className="px-3 py-1 bg-red-500 text-white rounded"
        >
          삭제하기
        </button>
        <button
          onClick={updateCurrentSetting}
          className="px-3 py-1 bg-green-600 text-white rounded"
          disabled={!selectedSettingName}
        >
          수정하기
        </button>
      </div>

      <div className="flex gap-4 items-end">
        <div className="flex-1">
          <label className="block font-semibold">📝 새로운 설정 이름</label>
          <input
            type="text"
            className="border p-2 rounded w-full"
            placeholder="예: 기본 공문 스타일"
            value={settingName}
            onChange={(e) => setSettingName(e.target.value)}
          />
        </div>
        <button
          onClick={saveNewSetting}
          className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          새로 저장하기
        </button>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <label className="block font-semibold">폰트</label>
          <select
            className="border p-2 rounded w-full"
            value={fontFamily}
            onChange={(e) => setFontFamily(e.target.value)}
          >
            <option value="Arial">Arial</option>
            <option value="Nanum Gothic">나눔고딕</option>
            <option value="Georgia">Georgia</option>
            <option value="sans-serif">Sans-serif</option>
          </select>
        </div>

        <div>
          <label className="block font-semibold">폰트 크기 (px)</label>
          <input
            type="number"
            className="border p-2 rounded w-full"
            value={fontSize}
            onChange={(e) => setFontSize(Number(e.target.value))}
          />
        </div>

        <div>
          <label className="block font-semibold">줄 간격 (line height)</label>
          <input
            type="number"
            step="0.1"
            className="border p-2 rounded w-full"
            value={lineHeight}
            onChange={(e) => setLineHeight(Number(e.target.value))}
          />
        </div>
      </div>


      <div>
        <label className="block font-semibold">🧷 앞에 들어갈 블록 (Prefix)</label>
        <SunEditor
          getSunEditorInstance={(inst) => { editorBeforeRef.current = inst; }}
          setContents={beforeBlock}
          onChange={setBeforeBlock}
          setOptions={{
            minHeight: '200',
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
      </div>

      <div>
        <label className="block font-semibold">🧷 뒤에 들어갈 블록 (Suffix)</label>
        <SunEditor
          getSunEditorInstance={(inst) => { editorAfterRef.current = inst; }}
          setContents={afterBlock}
          onChange={setAfterBlock}
          setOptions={{
            minHeight: '200',
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
      </div>

      <div className="mt-6 p-4 border rounded bg-gray-50">
        <h2 className="font-bold mb-2">미리보기</h2>
        <div
          className="border p-4 space-y-2"
          style={{ fontFamily, fontSize: `${fontSize}px`, lineHeight }}
          dangerouslySetInnerHTML={{ __html: `${beforeBlock}${afterBlock}` }}
        />
      </div>
    </div>
  );
}
