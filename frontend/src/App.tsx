import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { AitaPost } from './types'

type SearchMethod = 'SVD' | 'TF-IDF' | 'RAG'
type VerdictFilter = 'NTA' | 'YTA' | 'ESH' | 'NAH' | null

const VERDICT_COLORS: Record<string, string> = {
  NTA: '#2e7d32',
  YTA: '#c62828',
  ESH: '#e65100',
  NAH: '#1565c0',
}

const VERDICT_LABELS: Record<Exclude<VerdictFilter, null>, string> = {
  NTA: 'Not the Asshole',
  YTA: 'You’re the Asshole',
  ESH: 'Everyone Sucks Here',
  NAH: 'No Assholes Here',
}

interface RagState {
  rewrittenQuery: string | null
  answer: string
  loading: boolean
  step: 'idle' | 'rewriting' | 'retrieving' | 'answering' | 'done'
}

const RAG_IDLE: RagState = { rewrittenQuery: null, answer: '', loading: false, step: 'idle' }

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean>(true)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [posts, setPosts] = useState<AitaPost[]>([])
  const [method, setMethod] = useState<SearchMethod>('SVD')
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>(null)
  const [rag, setRag] = useState<RagState>(RAG_IDLE)
  const [errorMessage, setErrorMessage] = useState<string>('')

  useEffect(() => {
    fetch('/api/config')
      .then(r => r.json())
      .then(d => setUseLlm(Boolean(d.use_llm)))
      .catch(() => {
        // Keep the RAG button visible even if config can't be loaded yet.
        setUseLlm(true)
      })
  }, [])

  const runIrSearch = async (value: string, m: SearchMethod, vf: VerdictFilter) => {
    const mp = m === 'TF-IDF' ? 'tfidf' : 'svd'
    const vp = vf ? `&verdict=${encodeURIComponent(vf)}` : ''
    setErrorMessage('')

    try {
      const res = await fetch(`/api/search?query=${encodeURIComponent(value)}&method=${mp}${vp}`)
      if (!res.ok) {
        throw new Error(`Search request failed (${res.status})`)
      }

      const data = await res.json()
      setPosts(data)

      if (!Array.isArray(data) || data.length === 0) {
        setErrorMessage('No posts matched that search yet. Try a broader keyword.')
      }
    } catch {
      setPosts([])
      setErrorMessage('Search could not reach the backend. Make sure Flask is running on port 5001.')
    }
  }

  const runRagSearch = async (value: string) => {
    setRag({ ...RAG_IDLE, loading: true, step: 'rewriting' })
    setPosts([])
    setErrorMessage('')

    if (!useLlm) {
      setRag({
        ...RAG_IDLE,
        answer: 'RAG is available in the UI, but the backend does not have SPARK_API_KEY configured yet.',
        step: 'done',
      })
      return
    }

    try {
      const res = await fetch('/api/rag', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: value, method: 'svd' }),
      })

      if (!res.ok) {
        const d = await res.json()
        setRag({ ...RAG_IDLE, answer: 'Error: ' + (d.error || res.status), step: 'done' })
        return
      }

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value: chunk } = await reader.read()
        if (done) break
        buf += decoder.decode(chunk, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const ev = JSON.parse(line.slice(6))
            if (ev.rewritten_query !== undefined) {
              setRag(r => ({ ...r, rewrittenQuery: ev.rewritten_query, step: 'retrieving' }))
            } else if (ev.ir_results !== undefined) {
              setPosts(ev.ir_results)
              setRag(r => ({ ...r, step: 'answering' }))
            } else if (ev.content !== undefined) {
              setRag(r => ({ ...r, answer: r.answer + ev.content }))
            } else if (ev.done) {
              setRag(r => ({ ...r, loading: false, step: 'done' }))
            } else if (ev.error) {
              setRag(r => ({ ...r, loading: false, step: 'done', answer: r.answer + '\n[Error: ' + ev.error + ']' }))
            }
          } catch { /* malformed SSE line */ }
        }
      }
    } catch {
      setRag(r => ({ ...r, loading: false, step: 'done', answer: 'Connection error. Is the server running?' }))
    }
  }

  const handleSearch = async (
    value: string,
    m: SearchMethod = method,
    vf: VerdictFilter = verdictFilter
  ) => {
    setSearchTerm(value)
    if (value.trim() === '') {
      setPosts([])
      setRag(RAG_IDLE)
      setErrorMessage('')
      return
    }
    if (m === 'RAG') {
      await runRagSearch(value)
    } else {
      setRag(RAG_IDLE)
      await runIrSearch(value, m, vf)
    }
  }

  const handleMethodChange = (m: SearchMethod) => {
    setMethod(m)
    setRag(RAG_IDLE)
    if (searchTerm.trim()) handleSearch(searchTerm, m, verdictFilter)
  }

  const handleVerdictChange = (v: 'NTA' | 'YTA' | 'ESH' | 'NAH') => {
    const next: VerdictFilter = verdictFilter === v ? null : v
    setVerdictFilter(next)
    if (searchTerm.trim()) handleSearch(searchTerm, method, next)
  }

  const hasResults = posts.length > 0 || rag.step !== 'idle'

  const ragStepLabel: Record<RagState['step'], string> = {
    idle: '',
    rewriting: 'Step 1 — rewriting query…',
    retrieving: 'Step 2 — retrieving posts…',
    answering: 'Step 3 — synthesizing verdict…',
    done: '',
  }

  return (
    <div className={`full-body-container ${hasResults ? 'has-results' : ''}`}>
      <div className="main-content">

        {/* ── Header + search ── */}
        <div className="top-text">
          <h1 className="brain-rot-title">Brain rot</h1>
          <p className="tagline">Your daily dose of "Am I the Asshole?" — curated, narrated, and impossible to stop.</p>

          <div className="search-and-toggle">
            <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
              <img src={SearchIcon} alt="search" />
              <input
                id="search-input"
                placeholder={method === 'RAG' ? 'Describe your situation for an AI verdict…' : 'Search for an r/AITA post!'}
                value={searchTerm}
                onChange={e => {
                  const v = e.target.value
                  setSearchTerm(v)
                  if (method !== 'RAG') {
                    if (v.trim() === '') { setPosts([]); setRag(RAG_IDLE) }
                    else runIrSearch(v, method, verdictFilter)
                  }
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter') handleSearch(searchTerm)
                }}
              />
            </div>

            <div className="method-toggle">
              <span className="method-label">search method</span>
              <div className="toggle-buttons">
                <button className={method === 'SVD' ? 'active' : ''} onClick={() => handleMethodChange('SVD')}>SVD</button>
                <button className={method === 'TF-IDF' ? 'active' : ''} onClick={() => handleMethodChange('TF-IDF')}>TF-IDF</button>
                <button
                  className={method === 'RAG' ? 'active rag-btn' : 'rag-btn'}
                  onClick={() => handleMethodChange('RAG')}
                >
                  RAG
                </button>
              </div>
            </div>
          </div>

          {/* Verdict filter — only for IR methods */}
          {method !== 'RAG' && (
            <div className="verdict-filter-row">
              <span className="method-label">filter by verdict</span>
              <div className="verdict-buttons">
                {(['NTA', 'YTA', 'ESH', 'NAH'] as const).map(v => (
                  <button
                    key={v}
                    className={`verdict-btn${verdictFilter === v ? ' active' : ''}`}
                    style={verdictFilter === v ? { background: VERDICT_COLORS[v], color: '#fff', borderColor: VERDICT_COLORS[v] } : {}}
                    onClick={() => handleVerdictChange(v)}
                    title={VERDICT_LABELS[v]}
                    aria-label={`${v}: ${VERDICT_LABELS[v]}`}
                  >
                    {v}
                  </button>
                ))}
                {verdictFilter && (
                  <button className="verdict-btn clear-btn" onClick={() => { setVerdictFilter(null); if (searchTerm.trim()) handleSearch(searchTerm, method, null) }}>
                    All
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* ── Results area ── */}
        <div id="answer-box">
          {errorMessage && (
            <div className="search-error-card">
              {errorMessage}
            </div>
          )}

          {/* RAG step indicator */}
          {rag.loading && (
            <div className="rag-step-row">
              <div className="loading-indicator visible">
                <span className="loading-dot" /><span className="loading-dot" /><span className="loading-dot" />
              </div>
              <span className="rag-step-label">{ragStepLabel[rag.step]}</span>
            </div>
          )}

          {/* RAG: rewritten query pill */}
          {rag.rewrittenQuery && (
            <div className="rag-query-pill">
              <span className="rag-query-label">IR query</span>
              <span className="rag-query-text">{rag.rewrittenQuery}</span>
            </div>
          )}

          {/* RAG: LLM streaming answer — shown above posts so it's prominent */}
          {(rag.answer || rag.step === 'answering') && (
            <div className="llm-verdict-card">
              <div className="llm-verdict-header">AI Verdict</div>
              <p className="llm-verdict-body">
                {rag.answer}
                {rag.step === 'answering' && <span className="llm-cursor" />}
              </p>
            </div>
          )}

          {/* Retrieved / searched posts — same cards for all methods */}
          {posts.length > 0 && (
            <>
              {method === 'RAG' && (
                <div className="rag-ir-section-label">Retrieved posts used as context</div>
              )}
              {posts.map(post => (
                <div key={post.id} className="episode-item">
                  {post.verdict && (
                    <span className="verdict-badge" style={{ background: VERDICT_COLORS[post.verdict] ?? '#555' }}>
                      {post.verdict}
                    </span>
                  )}
                  <h3 className="episode-title">{post.title}</h3>
                  <p className="episode-desc">
                    {post.selftext ? post.selftext.slice(0, 400) : 'No text available'}
                    {post.selftext && post.selftext.length > 400 ? '...' : ''}
                  </p>
                  <p className="episode-rating">Net Votes: {post.score} · Similarity: {post.similarity?.toFixed(3)}</p>
                  {method === 'SVD' && post.svd_top_dimensions && post.svd_top_dimensions.length > 0 && (
                    <div className="svd-latent-tags" aria-label="Top latent SVD dimensions for this match">
                      <span className="svd-latent-tags-heading">Latent dimensions</span>
                      <div className="svd-latent-tags-row">
                        {post.svd_top_dimensions.map(dim => (
                          <span
                            key={`${post.id}-d${dim.dimension}`}
                            className="svd-latent-tag"
                            title={`post ${dim.post_value.toFixed(3)} · query ${dim.query_value.toFixed(3)} · contribution ${dim.contribution.toFixed(3)}`}
                          >
                            <span className="svd-latent-tag-dim">d{dim.dimension}</span>
                            <span className="svd-latent-tag-label">{dim.label}</span>
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
