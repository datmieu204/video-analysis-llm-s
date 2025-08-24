import { useState, forwardRef, useImperativeHandle } from 'react'
import { Youtube, Upload, FileVideo, Loader2, CheckCircle, ChevronDown, Play, BarChart3, Shield, Sparkles } from 'lucide-react'
import axios from 'axios'

const API_BASE_URL = 'http://127.0.0.1:8000/api'

const VideoProcessor = forwardRef(({ transcriptData, setTranscriptData, agentResults, setAgentResults }, ref) => {
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [uploadFile, setUploadFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [agentLoading, setAgentLoading] = useState({
    summary: false,
    highlights: false,
    violations: false,
    all: false
  })
  const [expandedSections, setExpandedSections] = useState({
    transcript: false,
    summary: true,
    highlights: false,
    violations: false
  })
  
  // Configuration states
  const [showYoutubeConfig, setShowYoutubeConfig] = useState(false)
  const [showUploadConfig, setShowUploadConfig] = useState(false)
  const [youtubeConfig, setYoutubeConfig] = useState({
    captions: true,
    provider: 'GROQ',
    model: 'llama-3.3-70b-versatile',
    language: 'en'
  })
  const [uploadConfig, setUploadConfig] = useState({
    language: 'en',
    provider: 'GROQ',
    model: 'llama-3.3-70b-versatile'
  })

  // YouTube processing
  const handleYoutubeSubmit = async (e) => {
    e.preventDefault()
    if (!youtubeUrl.trim()) return

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('url', youtubeUrl)
      formData.append('captions', youtubeConfig.captions.toString())
      formData.append('provider', youtubeConfig.provider)
      formData.append('model', youtubeConfig.model)
      formData.append('language', youtubeConfig.language)

      const response = await axios.post(`${API_BASE_URL}/transcript/youtube`, formData)
      setTranscriptData(response.data)
      setAgentResults({}) // Reset agent results for new transcript
    } catch (error) {
      console.error('Error processing YouTube video:', error)
      alert('Error processing YouTube video. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // File upload processing
  const handleFileSubmit = async (e) => {
    e.preventDefault()
    if (!uploadFile) return

    setLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', uploadFile)
      formData.append('language', uploadConfig.language)

      const response = await axios.post(`${API_BASE_URL}/transcript/upload`, formData)
      setTranscriptData(response.data)
      setAgentResults({}) // Reset agent results for new transcript
    } catch (error) {
      console.error('Error processing uploaded file:', error)
      alert('Error processing uploaded file. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault()
    setDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      setUploadFile(files[0])
    }
  }

  // Run AI agents
  const runAgents = async (taskTypes, agentKey = 'all') => {
    if (!transcriptData?.id) return

    setAgentLoading(prev => ({ ...prev, [agentKey]: true }))
    try {
      const formData = new FormData()
      taskTypes.forEach(task => formData.append('task_type', task))

      const response = await axios.post(
        `${API_BASE_URL}/transcript/agents/${transcriptData.id}`,
        formData
      )
      
      setAgentResults(prev => ({
        ...prev,
        ...response.data.results
      }))

      // Auto-expand the section if it has new data
      if (response.data.results.summary) {
        setExpandedSections(prev => ({ ...prev, summary: true }))
      }
      if (response.data.results.highlights) {
        setExpandedSections(prev => ({ ...prev, highlights: true }))
      }
      if (response.data.results.violations) {
        setExpandedSections(prev => ({ ...prev, violations: true }))
      }
    } catch (error) {
      console.error('Error running agents:', error)
      alert('Error running AI analysis. Please try again.')
    } finally {
      setAgentLoading(prev => ({ ...prev, [agentKey]: false }))
    }
  }

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  // Expose runAgents function to parent component
  useImperativeHandle(ref, () => ({
    runAgents
  }), [transcriptData])

  if (loading) {
    return (
      <div className="video-processor full-height">
        <div className="loading">
          <Loader2 size={48} className="spin" />
          <h3>Processing your video...</h3>
          <p>This process may take a few minutes depending on video length.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="video-processor full-height">
      <div className="processor-header">
        <h2>Video Processing</h2>
        <p>Upload a video file or provide a YouTube URL to get started</p>
      </div>

      <div className="input-methods">
        {/* YouTube URL Input */}
        <div className="input-method">
          <h3>
            <Youtube size={20} />
            YouTube Video
          </h3>
          <form onSubmit={handleYoutubeSubmit}>
            <div className="form-group">
              <label htmlFor="youtube-url">YouTube URL</label>
              <input
                id="youtube-url"
                type="url"
                className="form-input"
                placeholder="https://www.youtube.com/watch?v=..."
                value={youtubeUrl}
                onChange={(e) => setYoutubeUrl(e.target.value)}
                required
              />
            </div>
            
            <button 
              type="button" 
              className="config-toggle"
              onClick={() => setShowYoutubeConfig(!showYoutubeConfig)}
            >
              {showYoutubeConfig ? 'Hide Configuration' : 'Show Configuration'}
            </button>
            
            {showYoutubeConfig && (
              <div className="config-section">
                <div className="config-grid">
                  <div className="form-group">
                    <label>Provider</label>
                    <select 
                      className="form-select"
                      value={youtubeConfig.provider}
                      onChange={(e) => setYoutubeConfig({...youtubeConfig, provider: e.target.value})}
                    >
                      <option value="GROQ">GROQ</option>
                      <option value="OpenAI">OpenAI</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Model</label>
                    <select 
                      className="form-select"
                      value={youtubeConfig.model}
                      onChange={(e) => setYoutubeConfig({...youtubeConfig, model: e.target.value})}
                    >
                      <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                      <option value="gpt-4">GPT-4</option>
                      <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Language</label>
                    <select 
                      className="form-select"
                      value={youtubeConfig.language}
                      onChange={(e) => setYoutubeConfig({...youtubeConfig, language: e.target.value})}
                    >
                      <option value="en">English</option>
                      <option value="vi">Vietnamese</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>
                      <input 
                        type="checkbox"
                        checked={youtubeConfig.captions}
                        onChange={(e) => setYoutubeConfig({...youtubeConfig, captions: e.target.checked})}
                        style={{marginRight: '0.5rem'}}
                      />
                      Use YouTube Captions
                    </label>
                  </div>
                </div>
              </div>
            )}
            
            <button type="submit" className="submit-button" disabled={loading}>
              <Youtube size={16} />
              Process YouTube Video
            </button>
          </form>
        </div>

        {/* File Upload */}
        <div className="input-method">
          <h3>
            <Upload size={20} />
            Upload File
          </h3>
          <form onSubmit={handleFileSubmit}>
            <div className="form-group">
              <div
                className={`file-upload ${dragOver ? 'dragover' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById('file-input').click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept=".mp3,.mp4,.wav,.m4a,.webm"
                  onChange={(e) => setUploadFile(e.target.files[0])}
                  style={{ display: 'none' }}
                />
                {uploadFile ? (
                  <div>
                    <FileVideo size={32} />
                    <p><strong>{uploadFile.name}</strong></p>
                    <p>File selected successfully</p>
                  </div>
                ) : (
                  <div>
                    <Upload size={32} />
                    <p>Drag and drop your video file here</p>
                    <p>or click to browse</p>
                    <small>Supports: MP3, MP4, WAV, M4A, WebM</small>
                  </div>
                )}
              </div>
            </div>
            
            <button 
              type="button" 
              className="config-toggle"
              onClick={() => setShowUploadConfig(!showUploadConfig)}
            >
              {showUploadConfig ? 'Hide Configuration' : 'Show Configuration'}
            </button>
            
            {showUploadConfig && (
              <div className="config-section">
                <div className="config-grid">
                  <div className="form-group">
                    <label>Provider</label>
                    <select 
                      className="form-select"
                      value={uploadConfig.provider}
                      onChange={(e) => setUploadConfig({...uploadConfig, provider: e.target.value})}
                    >
                      <option value="GROQ">GROQ</option>
                      <option value="OpenAI">OpenAI</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Model</label>
                    <select 
                      className="form-select"
                      value={uploadConfig.model}
                      onChange={(e) => setUploadConfig({...uploadConfig, model: e.target.value})}
                    >
                      <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                      <option value="gpt-4">GPT-4</option>
                      <option value="gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label>Language</label>
                    <select 
                      className="form-select"
                      value={uploadConfig.language}
                      onChange={(e) => setUploadConfig({...uploadConfig, language: e.target.value})}
                    >
                      <option value="en">English</option>
                      <option value="vi">Vietnamese</option>
                      <option value="es">Spanish</option>
                      <option value="fr">French</option>
                    </select>
                  </div>
                </div>
              </div>
            )}
            
            <button type="submit" className="submit-button" disabled={!uploadFile || loading}>
              <Upload size={16} />
              Process Upload
            </button>
          </form>
        </div>
      </div>

      {/* Results Section */}
      {transcriptData && (
        <div className="results">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
            <CheckCircle size={20} color="green" />
            <h3 style={{ color: '#333' }}>Transcript Completed</h3>
          </div>
          
          <div className="form-group">
            <button 
              type="button" 
              className="config-toggle"
              onClick={() => setExpandedSections(prev => ({ ...prev, transcript: !prev.transcript }))}
              style={{ marginBottom: '0.5rem' }}
            >
              {expandedSections.transcript ? 'Hide Transcript' : 'Show Transcript Preview'}
            </button>
            {expandedSections.transcript && (
              <textarea
                className="form-input"
                value={transcriptData.transcript?.substring(0, 400) + '...'}
                readOnly
                rows={4}
                style={{ resize: 'none', fontSize: '0.9rem' }}
              />
            )}
          </div>

          {/* Agent Controls */}
          <div className="agent-controls">
            <h4 style={{ width: '100%', marginBottom: '0.5rem', color: '#333' }}>AI Analysis Tools:</h4>
            <button
              className="agent-button"
              onClick={() => runAgents(['summarize'], 'summary')}
              disabled={agentLoading.summary}
            >
              {agentLoading.summary ? <Loader2 size={14} className="spin" /> : <BarChart3 size={14} />}
              Summary
            </button>
            <button
              className="agent-button"
              onClick={() => runAgents(['highlight'], 'highlights')}
              disabled={agentLoading.highlights}
            >
              {agentLoading.highlights ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} />}
              Highlights
            </button>
            <button
              className="agent-button"
              onClick={() => runAgents(['violation'], 'violations')}
              disabled={agentLoading.violations}
            >
              {agentLoading.violations ? <Loader2 size={14} className="spin" /> : <Shield size={14} />}
              Violations
            </button>
            <button
              className="agent-button"
              onClick={() => runAgents(['summarize', 'highlight', 'violation'], 'all')}
              disabled={agentLoading.all}
              style={{ background: '#28a745' }}
            >
              {agentLoading.all ? <Loader2 size={14} className="spin" /> : <Play size={14} />}
              Run All
            </button>
          </div>

          {/* Agent Results */}
          <div className="agent-results">
            {/* Summary Section */}
            {agentResults.summary && (
              <div className="agent-result-section">
                <div 
                  className="agent-result-header"
                  onClick={() => toggleSection('summary')}
                >
                  <div className="agent-result-title">
                    <BarChart3 size={16} />
                    Summary
                  </div>
                  <ChevronDown 
                    size={16} 
                    className={`agent-result-toggle ${expandedSections.summary ? 'expanded' : ''}`}
                  />
                </div>
                <div className={`agent-result-content ${expandedSections.summary ? '' : 'collapsed'}`}>
                  <div className="summary-content">{agentResults.summary}</div>
                </div>
              </div>
            )}

            {/* Highlights Section */}
            {agentResults.highlights && agentResults.highlights.length > 0 && (
              <div className="agent-result-section">
                <div 
                  className="agent-result-header"
                  onClick={() => toggleSection('highlights')}
                >
                  <div className="agent-result-title">
                    <Sparkles size={16} />
                    Highlights ({agentResults.highlights.length})
                  </div>
                  <ChevronDown 
                    size={16} 
                    className={`agent-result-toggle ${expandedSections.highlights ? 'expanded' : ''}`}
                  />
                </div>
                <div className={`agent-result-content ${expandedSections.highlights ? '' : 'collapsed'}`}>
                  <div className="highlights-list">
                    {agentResults.highlights.map((highlight, index) => (
                      <div key={index} className="highlight-item">
                        <div className="highlight-timestamp">{highlight.timestamp}</div>
                        <div className="highlight-text">{highlight.text}</div>
                        {highlight.reason && (
                          <div className="highlight-reason">
                            <strong>Reason:</strong> {highlight.reason}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Violations Section */}
            {agentResults.violations && agentResults.violations.length > 0 && (
              <div className="agent-result-section">
                <div 
                  className="agent-result-header"
                  onClick={() => toggleSection('violations')}
                >
                  <div className="agent-result-title">
                    <Shield size={16} />
                    Content Violations ({agentResults.violations.length})
                  </div>
                  <ChevronDown 
                    size={16} 
                    className={`agent-result-toggle ${expandedSections.violations ? 'expanded' : ''}`}
                  />
                </div>
                <div className={`agent-result-content ${expandedSections.violations ? '' : 'collapsed'}`}>
                  <div className="violations-list">
                    {agentResults.violations.map((violation, index) => (
                      <div key={index} className="violation-item">
                        <div className="violation-type">{violation.violation}</div>
                        <div className="violation-explanation">{violation.explanation}</div>
                        {violation.timestamp && (
                          <div className="violation-timestamp">Timestamp: {violation.timestamp}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <style jsx>{`
        .spin {
          animation: spin 1s linear infinite;
        }
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
})

export default VideoProcessor
