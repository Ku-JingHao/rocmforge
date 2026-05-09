import Editor from '@monaco-editor/react';

interface Props {
  value: string;
  onChange?: (value: string) => void;
  language: string;
  readOnly?: boolean;
  height?: string;
}

const languageMap: Record<string, string> = {
  pytorch: 'python',
  cuda: 'cpp',
  triton: 'python',
  english: 'plaintext',
  hip: 'cpp',
  cpp: 'cpp',
};

export function CodeEditor({ value, onChange, language, readOnly = false, height = '500px' }: Props) {
  const monacoLang = languageMap[language] || 'plaintext';

  return (
    <Editor
      height={height}
      language={monacoLang}
      theme="vs-dark"
      value={value}
      onChange={(v) => onChange?.(v ?? '')}
      options={{
        readOnly,
        fontSize: 13,
        fontFamily: 'JetBrains Mono, Menlo, monospace',
        minimap: { enabled: false },
        scrollBeyondLastLine: false,
        wordWrap: 'on',
        lineNumbers: 'on',
        tabSize: 2,
        renderLineHighlight: 'gutter',
        smoothScrolling: true,
      }}
    />
  );
}
