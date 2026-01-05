
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { 
  FileText, 
  Send, 
  Trash2, 
  MessageSquare, 
  Loader2, 
  Plus,
  FileQuestion,
  Info,
  ChevronRight,
  Sparkles,
  Paperclip,
  Settings,
  Key,
  Database,
  Cpu,
  Zap,
  ShieldCheck
} from 'lucide-react';

// Types
type Provider = 'gemini' | 'openai' | 'anthropic';

interface DocFile {
  id: string;
  name: string;
  type: string;
  base64: string;
  size: string;
}

interface ChatMessage {
  role: 'user' | 'model';
  content: string;
  isStreaming?: boolean;
}

const formatSize = (bytes: number) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

const App: React.FC = () => {
  const [provider, setProvider] = useState<Provider>('gemini');
  const [docs, setDocs] = useState<DocFile[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('gemini-3-pro-preview');
  
  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Theme configuration
  const theme = useMemo(() => {
    switch (provider) {
      case 'openai':
        return {
          brand: 'OpenAI',
          primary: 'emerald',
          bg: 'bg-zinc-950',
          sidebar: 'bg-emerald-950/20',
          accent: 'text-emerald-400',
          button: 'bg-emerald-500 hover:bg-emerald-400 text-zinc-950',
          border: 'border-emerald-900/30',
          msgUser: 'bg-emerald-600 text-white',
          msgBot: 'bg-zinc-900 border-emerald-900/40 text-emerald-50',
          icon: <Zap size={20} />
        };
      case 'anthropic':
        return {
          brand: 'Anthropic',
          primary: 'orange',
          bg: 'bg-stone-950',
          sidebar: 'bg-stone-900',
          accent: 'text-orange-300',
          button: 'bg-orange-200 hover:bg-white text-stone-950',
          border: 'border-stone-800',
          msgUser: 'bg-stone-200 text-stone-950',
          msgBot: 'bg-stone-900 border-stone-800 text-stone-100',
          icon: <ShieldCheck size={20} />
        };
      default: // gemini
        return {
          brand: 'Google',
          primary: 'zinc',
          bg: 'bg-zinc-950',
          sidebar: 'bg-zinc-900',
          accent: 'text-zinc-100',
          button: 'bg-zinc-100 hover:bg-white text-zinc-950',
          border: 'border-zinc-800',
          msgUser: 'bg-zinc-100 text-zinc-950',
          msgBot: 'bg-zinc-900 border-zinc-800 text-zinc-100',
          icon: <Sparkles size={20} />
        };
    }
  }, [provider]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Update models when provider changes
  useEffect(() => {
    if (provider === 'openai') setSelectedModel('gpt-4o');
    else if (provider === 'anthropic') setSelectedModel('claude-3-5-sonnet');
    else setSelectedModel('gemini-3-pro-preview');
  }, [provider]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;
    setIsUploading(true);
    let processedCount = 0;
    const totalFiles = files.length;
    Array.from(files).forEach((file: File) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        if (event.target?.result) {
          const base64 = (event.target.result as string).split(',')[1];
          const newDoc: DocFile = {
            id: Math.random().toString(36).substr(2, 9),
            name: file.name,
            type: file.type,
            base64,
            size: formatSize(file.size)
          };
          setDocs(prev => [...prev, newDoc]);
          processedCount++;
          if (processedCount === totalFiles) setIsUploading(false);
        }
      };
      reader.readAsDataURL(file);
    });
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const removeDoc = (id: string) => {
    setDocs(prev => prev.filter(d => d.id !== id));
  };

  const openKeySelector = async () => {
    if (window.aistudio?.openSelectKey) {
      await window.aistudio.openSelectKey();
    } else {
      alert(`Please set your API key for ${provider.toUpperCase()} in your environment configuration.`);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || isProcessing) return;

    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setIsProcessing(true);

    try {
      if (provider === 'gemini') {
        // Create a new GoogleGenAI instance right before making an API call
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const parts: any[] = docs.map(doc => ({
          inlineData: { data: doc.base64, mimeType: doc.type || 'text/plain' }
        }));
        parts.push({ text: `Context: Analyze the files provided. Question: ${userMessage}` });

        const streamResponse = await ai.models.generateContentStream({
          model: selectedModel,
          contents: { parts },
          config: {
            systemInstruction: "You are a professional document assistant. Answer based on the provided files. Use markdown.",
            temperature: 0.1,
          },
        });

        let fullResponse = '';
        setMessages(prev => [...prev, { role: 'model', content: '', isStreaming: true }]);

        for await (const chunk of streamResponse) {
          // Use .text property directly
          if (chunk.text) {
            fullResponse += chunk.text;
            setMessages(prev => {
              const newMsgs = [...prev];
              newMsgs[newMsgs.length - 1].content = fullResponse;
              return newMsgs;
            });
          }
        }
      } else {
        // Mocking External API response for Demo purposes 
        // In a real app, you would use fetch() to OpenAI/Anthropic/OpenRouter endpoints here
        setMessages(prev => [...prev, { role: 'model', content: '', isStreaming: true }]);
        const mockResponse = `This is a simulated response from ${theme.brand} using ${selectedModel}. \n\nDirect browser calls to ${provider} API are typically blocked by CORS. To use this in production, you would connect this frontend to an OpenRouter endpoint or your own Node.js proxy server.`;
        
        // Simulate streaming
        let current = '';
        for (const char of mockResponse.split(' ')) {
          await new Promise(r => setTimeout(r, 40));
          current += char + ' ';
          setMessages(prev => {
            const newMsgs = [...prev];
            newMsgs[newMsgs.length - 1].content = current;
            return newMsgs;
          });
        }
      }

      setMessages(prev => {
        const newMsgs = [...prev];
        newMsgs[newMsgs.length - 1].isStreaming = false;
        return newMsgs;
      });

    } catch (error: any) {
      console.error("API Error:", error);
      setMessages(prev => [...prev, { role: 'model', content: "Error connecting to provider. Check API key and CORS settings." }]);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className={`flex h-screen w-full ${theme.bg} text-zinc-100 overflow-hidden selection:bg-zinc-700 transition-colors duration-700`}>
      {/* Sidebar */}
      <aside className={`w-80 ${theme.sidebar} border-r ${theme.border} flex flex-col z-10 transition-colors duration-700`}>
        <div className={`p-6 border-b ${theme.border}`}>
          <div className="flex items-center gap-3 mb-1">
            <div className={`p-2 rounded-lg ${theme.button} shadow-xl transition-all duration-500`}>
              {theme.icon}
            </div>
            <h1 className="text-xl font-black tracking-tighter text-white uppercase italic">
              QueryDocs
            </h1>
          </div>
          <p className="text-[10px] text-zinc-500 font-bold uppercase tracking-[0.2em]">{provider} Powered</p>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-6">
          <section>
            <div className="flex items-center justify-between mb-4 px-2">
              <h2 className="text-[10px] font-black text-zinc-500 flex items-center gap-2 uppercase tracking-[0.2em]">
                <FileText size={12} /> Intelligence context
              </h2>
              <span className={`text-[10px] font-bold bg-zinc-800 ${theme.accent} px-2 py-0.5 rounded-full border ${theme.border}`}>
                {docs.length}
              </span>
            </div>
            
            <div className="space-y-2">
              {docs.map((doc) => (
                <div key={doc.id} className={`group relative flex items-center gap-3 p-3 bg-zinc-950/40 border ${theme.border} rounded-xl hover:bg-zinc-800/20 transition-all duration-200`}>
                  <div className={`p-2 rounded-lg bg-zinc-900 ${theme.accent}`}>
                    <FileText size={16} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-bold text-zinc-200 truncate">{doc.name}</p>
                    <p className="text-[9px] text-zinc-500 font-mono uppercase">{doc.size} • {doc.type.split('/')[1] || 'FILE'}</p>
                  </div>
                  <button onClick={() => removeDoc(doc.id)} className="opacity-0 group-hover:opacity-100 p-1.5 text-zinc-500 hover:text-white transition-all">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}

              {docs.length === 0 && (
                <div className={`text-center py-10 px-4 border border-dashed ${theme.border} rounded-2xl opacity-40`}>
                  <FileQuestion className="mx-auto mb-2" size={32} />
                  <p className="text-[10px] font-bold uppercase tracking-widest">Feed the model.</p>
                </div>
              )}
            </div>
          </section>

          <section>
             <button 
                onClick={() => fileInputRef.current?.click()}
                className={`w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-black text-xs uppercase tracking-widest transition-all shadow-lg active:scale-95 ${theme.button}`}
              >
                <Plus size={16} /> Add Context
              </button>
              <input type="file" className="hidden" ref={fileInputRef} onChange={handleFileUpload} multiple accept=".pdf,.txt,.md,.png,.jpg,.jpeg" />
          </section>

          <section className={`pt-6 border-t ${theme.border}`}>
            <h2 className="text-[10px] font-black text-zinc-500 flex items-center gap-2 uppercase tracking-[0.2em] mb-4 px-2">
              <Settings size={12} /> Neural Engines
            </h2>
            <div className="space-y-2 px-1">
              {[
                { id: 'gemini', name: 'Google Gemini', icon: <Sparkles size={14}/> },
                { id: 'openai', name: 'OpenAI GPT-4', icon: <Zap size={14}/> },
                { id: 'anthropic', name: 'Anthropic Claude', icon: <ShieldCheck size={14}/> }
              ].map((p) => (
                <button 
                  key={p.id}
                  onClick={() => setProvider(p.id as Provider)}
                  className={`w-full flex items-center gap-3 p-3 rounded-xl border transition-all ${provider === p.id ? `bg-zinc-800 ${theme.border} shadow-inner` : 'bg-transparent border-transparent hover:bg-zinc-800/40'}`}
                >
                  <div className={`p-1.5 rounded-lg ${provider === p.id ? theme.button : 'bg-zinc-900 text-zinc-600'}`}>
                    {p.icon}
                  </div>
                  <p className={`text-[11px] font-bold uppercase tracking-wider ${provider === p.id ? 'text-zinc-100' : 'text-zinc-500'}`}>{p.name}</p>
                </button>
              ))}

              <button 
                onClick={openKeySelector}
                className={`w-full flex items-center justify-between gap-2 py-3 px-4 mt-4 bg-zinc-900/50 border ${theme.border} rounded-xl text-[10px] font-black uppercase tracking-widest text-zinc-500 hover:text-white transition-all`}
              >
                <div className="flex items-center gap-2">
                  <Key size={12} />
                  <span>Configure Keys</span>
                </div>
                <ChevronRight size={12} />
              </button>
            </div>
          </section>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col relative">
        <header className={`h-16 border-b ${theme.border} flex items-center justify-between px-8 bg-black/20 backdrop-blur-xl sticky top-0 z-10`}>
          <div className="flex items-center gap-3">
            <Cpu size={16} className={theme.accent} />
            <span className={`text-[11px] font-black tracking-[0.2em] uppercase ${theme.accent}`}>{provider} Active Stream</span>
            {isProcessing && (
              <span className={`flex items-center gap-2 text-[10px] font-black animate-pulse-subtle px-3 py-1 rounded-full border ${theme.border} bg-black/40`}>
                <Loader2 size={10} className="animate-spin" /> ANALYZING...
              </span>
            )}
          </div>
          <div className="flex -space-x-2">
            {docs.slice(0, 3).map(d => (
                <div key={d.id} className={`w-7 h-7 rounded-full bg-zinc-900 border ${theme.border} flex items-center justify-center text-zinc-500 shadow-2xl`}>
                  <FileText size={10} />
                </div>
            ))}
          </div>
        </header>

        <div className="flex-1 overflow-y-auto p-8 custom-scrollbar space-y-10 max-w-4xl mx-auto w-full">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center space-y-8 pt-10">
              <div className={`p-10 rounded-[3rem] ${theme.sidebar} shadow-2xl border ${theme.border} transition-all duration-700`}>
                {/* Fixed TypeScript error by casting to React.ReactElement<any> to allow 'size' prop */}
                {React.cloneElement(theme.icon as React.ReactElement<any>, { size: 64 })}
              </div>
              <div className="space-y-4">
                <h3 className="text-4xl font-black text-white tracking-tighter uppercase italic drop-shadow-2xl">
                  {theme.brand} Analysis
                </h3>
                <p className="text-zinc-500 max-w-sm mx-auto text-xs font-bold uppercase tracking-widest leading-loose">
                  Select your engine. Upload your knowledge. <br/>
                  Ask anything.
                </p>
              </div>
            </div>
          ) : (
            messages.map((msg, idx) => (
              <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-500`}>
                <div className={`flex gap-5 max-w-[90%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 shadow-2xl border ${theme.border} ${msg.role === 'user' ? theme.button : 'bg-zinc-900 text-white'}`}>
                    {msg.role === 'user' ? <span className="text-[10px] font-black">USER</span> : /* Fixed TypeScript error by casting to React.ReactElement<any> */ React.cloneElement(theme.icon as React.ReactElement<any>, { size: 16 })}
                  </div>
                  <div className={`p-6 rounded-[2rem] shadow-2xl border ${msg.role === 'user' ? `${theme.msgUser} rounded-tr-none border-transparent` : `${theme.msgBot} rounded-tl-none`}`}>
                    <div className="prose prose-sm max-w-none whitespace-pre-wrap font-bold text-[13px] leading-relaxed tracking-tight">
                      {msg.content}
                      {msg.isStreaming && (
                         <span className={`inline-block w-2 h-4 ${theme.accent} ml-1 animate-pulse rounded-full align-middle`}></span>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={chatEndRef} />
        </div>

        {/* Input */}
        <div className="p-8">
          <div className="max-w-4xl mx-auto">
            <div className={`relative glass-panel rounded-[2.5rem] shadow-2xl border ${theme.border} overflow-hidden flex flex-col p-3 transition-colors duration-700`}>
              <div className="flex items-center gap-3 px-4 pb-2">
                <textarea 
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
                  placeholder={`Consult ${theme.brand} intelligence...`}
                  className="flex-1 bg-transparent border-none focus:ring-0 text-white placeholder:text-zinc-700 text-sm py-4 px-2 resize-none max-h-40 min-h-[60px] custom-scrollbar font-bold"
                  rows={1}
                />
                <div className="flex items-center gap-3 pr-2">
                  <button onClick={() => fileInputRef.current?.click()} className="p-4 text-zinc-600 hover:text-white transition-all">
                    <Paperclip size={22} />
                  </button>
                  <button 
                    disabled={!input.trim() || isProcessing || docs.length === 0}
                    onClick={sendMessage}
                    className={`p-5 rounded-3xl shadow-2xl transition-all active:scale-90 disabled:opacity-20 ${theme.button}`}
                  >
                    {isProcessing ? <Loader2 size={22} className="animate-spin" /> : <Send size={22} />}
                  </button>
                </div>
              </div>
            </div>
            <div className="flex items-center justify-center gap-6 mt-8">
               <div className={`h-[1px] flex-1 bg-zinc-900`}></div>
               <p className="text-[9px] text-zinc-700 font-black uppercase tracking-[0.4em]">
                 Cross-Model RAG Architecture v2.0
               </p>
               <div className={`h-[1px] flex-1 bg-zinc-900`}></div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

createRoot(document.getElementById('root')!).render(<App />);
