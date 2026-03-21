import { useState, useEffect } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { AitaPost } from './types'
import Chat from './Chat'

function App(): JSX.Element {
  const [useLlm, setUseLlm] = useState<boolean | null>(null)
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [posts, setPosts] = useState<AitaPost[]>([])

  useEffect(() => {
    fetch('/api/config').then(r => r.json()).then(data => setUseLlm(data.use_llm))
  }, [])

  const handleSearch = async (value: string): Promise<void> => {
  setSearchTerm(value)
  if (value.trim() === '') {
    setPosts([])
    return
  }

  const response = await fetch(`/api/search?query=${encodeURIComponent(value)}`)
  const data: AitaPost[] = await response.json()
  setPosts(data)
}

  if (useLlm === null) return <></>

  return (
   <div className={`full-body-container ${useLlm ? 'llm-mode' : ''} ${posts.length > 0 ? 'has-results' : ''}`}>
      {/* Search bar (always shown) */}
      <div className="top-text">
        <h1 className="brain-rot-title">Brian rot</h1>
        <p className="tagline">Your daily dose of "Am I the Asshole?" — curated, narrated, and impossible to stop.</p>
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search for an r/AITA post!"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
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
      <p className="episode-rating">Score: {post.score}</p>
    </div>
  ))}
</div>

      {/* Chat (only when USE_LLM = True in routes.py) */}
      {useLlm && <Chat onSearchTerm={handleSearch} />}
    </div>
  )
}

export default App
