import { useState, useRef } from 'react'
import './App.css'
import VideoProcessor from './components/VideoProcessor'
import Chatbot from './components/Chatbot'

function App() {
  const [transcriptData, setTranscriptData] = useState(null)
  const [agentResults, setAgentResults] = useState({})

  const videoProcessorRef = useRef(null)

  const handleRunAgent = async (taskTypes, agentKey) => {
    if (videoProcessorRef.current && videoProcessorRef.current.runAgents) {
      return await videoProcessorRef.current.runAgents(taskTypes, agentKey)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Video Analysis AI</h1>
        <p>Analyze videos with AI-powered transcription, summarization, and content moderation</p>
      </header>

      <main className="main-content-full">
        <div className="content-grid">
          <div className="left-panel">
            <VideoProcessor 
              ref={videoProcessorRef}
              transcriptData={transcriptData}
              setTranscriptData={setTranscriptData}
              agentResults={agentResults}
              setAgentResults={setAgentResults}
            />
          </div>
          <div className="right-panel">
            <Chatbot 
              transcriptData={transcriptData}
              agentResults={agentResults}
              onRunAgent={handleRunAgent}
            />
          </div>
        </div>
      </main>
    </div>
  )
}

export default App
