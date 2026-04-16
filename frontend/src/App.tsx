import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { AitaPost } from './types'
import Chat from './Chat'

type SearchMethod = 'SVD' | 'TF-IDF'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [posts, setPosts] = useState<AitaPost[]>([])
  const [method, setMethod] = useState<SearchMethod>('SVD')

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = async (value: string, searchMethod: SearchMethod = method): Promise<void> => {
    setSearchTerm(value)
    if (value.trim() === '') {
      setPosts([])
      return
    }
    const methodParam = searchMethod === 'TF-IDF' ? 'tfidf' : 'svd'
    const response = await fetch(
      `/api/search?query=${encodeURIComponent(value)}&method=${methodParam}`
    )
    const data: AitaPost[] = await response.json()
    setPosts(data)
  }

  const handleMethodChange = (newMethod: SearchMethod) => {
   setMethod(newMethod)
    if (searchTerm.trim() !== '') handleSearch(searchTerm, newMethod)
  

}
  

  if (useLlm === null) return <></>

  return (
   <div className={`full-body-container ${useLlm ? 'llm-mode' : ''} ${posts.length > 0 ? 'has-results' : ''}`}>
      {/* Search bar (always shown) */}
      <div className="top-text">
        <h1 className="brain-rot-title">Brian rot</h1>
        <p className="tagline">Your daily dose of "Am I the Asshole?" — curated, narrated, and impossible to stop.</p>
        
        <div className="search-and-toggle">
          <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
            <img src={SearchIcon} alt="search" />
            <input
              id="search-input"
              placeholder="Search for an r/AITA post!"
              value={searchTerm}
              onChange={(e) => handleSearch(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(searchTerm) }}
          />
        </div>
        <div className="method-toggle">
          <span className="method-label">search method</span>
          <div className="toggle-buttons">
            <button className={method === 'SVD' ? 'active' : ''} onClick={() => handleMethodChange('SVD')}>SVD</button>
            <button className={method === 'TF-IDF' ? 'active' : ''} onClick={() => handleMethodChange('TF-IDF')}>TF-IDF</button>
          </div>
        </div>
        </div>
      </div>

      {/* Search results (always shown) */}
      <div id="answer-box">
  {posts.map((post) => (
    <div key={post.id} className="episode-item">
      <h3 className="episode-title">{post.title}</h3>
      <p className="episode-desc">
        {post.selftext ? post.selftext.slice(0, 400) : 'No text available'}
        {post.selftext && post.selftext.length > 400 ? '...' : ''}
      </p>
      <p className="episode-rating">Score: {post.score} · similarity: {post.similarity?.toFixed(3)}</p>
    </div>
  ))}
</div>

      {/* Chat (only when USE_LLM = True in routes.py) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App
