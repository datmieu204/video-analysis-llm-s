import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2, MessageCircle, Database, Brain } from 'lucide-react'
import chatService from '../services/chatService'

const Chatbot = ({ transcriptData, agentResults, onRunAgent }) => {
  const [messages, setMessages] = useState([
    {
      id: 1,
      type: 'bot',
      content: 'Hello! I\'m your AI assistant. I can help you analyze and discuss the processed transcript. Feel free to ask questions about the content, or type keywords like "summary", "highlights", or "violations" to automatically run analysis!',
      timestamp: new Date().toLocaleTimeString(),
      intent: 'system'
    }
  ])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isTranscriptInVectorStore, setIsTranscriptInVectorStore] = useState(false)
  const [vectorStoreStatus, setVectorStoreStatus] = useState('idle') // idle, adding, added, error
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    const addTranscriptToVectorStore = async () => {
      if (transcriptData?.id && !isTranscriptInVectorStore && vectorStoreStatus === 'idle') {
        setVectorStoreStatus('adding')
        try {
          const result = await chatService.addTranscriptToVectorStore(transcriptData.id)
          if (result.success) {
            setIsTranscriptInVectorStore(true)
            setVectorStoreStatus('added')
            
            const systemMessage = {
              id: Date.now(),
              type: 'bot',
              content: '✅ Transcript has been added to the vector database. You can now ask questions related to the transcript content.',
              timestamp: new Date().toLocaleTimeString(),
              intent: 'system'
            }
            setMessages(prev => [...prev, systemMessage])
          }
        } catch (error) {
          console.error('Error adding transcript to vector store:', error)
          setVectorStoreStatus('error')
          
          const errorMessage = {
            id: Date.now(),
            type: 'bot',
            content: '❌ Error adding transcript to vector database.',
            timestamp: new Date().toLocaleTimeString(),
            intent: 'system'
          }
          setMessages(prev => [...prev, errorMessage])
        }
      }
    }

    addTranscriptToVectorStore()
  }, [transcriptData?.id, isTranscriptInVectorStore, vectorStoreStatus])

  // Intent classification
  const handleUserMessage = async (userMessage) => {
    const intent = chatService.classifyIntent(userMessage)
    console.log('Intent classification:', intent) // Debug log
    
    let botResponse = ''
    let responseIntent = intent.type

    try {
      switch (intent.type) {
        case 'agent_automation':
          // Agent automation
          botResponse = await handleAgentAutomation(intent.agentType, userMessage)
          break
          
        case 'rag_question':
          // RAG question
          console.log('RAG question detected. Vector store status:', isTranscriptInVectorStore) // Debug log
          if (isTranscriptInVectorStore) {
            const ragResponse = await chatService.askWithRAG(userMessage, true)
            console.log('Received RAG response:', ragResponse) // Debug log
            botResponse = ragResponse.response || 'Sorry, I couldn\'t find any relevant information in the transcript.'
          } else {
            botResponse = 'Transcript has not been added to the vector database yet. Please wait for the processing to complete.'
          }
          break
          
        case 'general_chat':
        default:
          // General chat
          const chatResponse = await chatService.askQuestion(userMessage, true)
          console.log('Received chat response:', chatResponse) // Debug log
          botResponse = chatResponse.response || 'Sorry, I encountered an error while processing your request. Please try again.'
          break
      }
    } catch (error) {
      console.error('Error handling user message:', error)
      botResponse = 'Sorry, I encountered an error while processing your request. Please try again.'
      responseIntent = 'error'
    }

    return { content: botResponse, intent: responseIntent }
  }

  // Agent automation
  const handleAgentAutomation = async (agentType, userMessage) => {
    if (!transcriptData?.id || !onRunAgent) {
      return 'Please upload the transcript before using the analysis feature.'
    }

    const agentMap = {
      'summarize': 'summary',
      'highlight': 'highlights', 
      'violation': 'violations'
    }

    const resultKey = agentMap[agentType]

    // Check if agent has already run
    if (agentResults?.[resultKey]) {
      return await generateAgentBasedResponse(agentType, agentResults[resultKey])
    }

    // Run agent if no results found
    try {
      await onRunAgent([agentType], resultKey)
      return getAgentRunningMessage(agentType)
    } catch (error) {
      console.error('Error running agent:', error)
      return `Error running ${agentType} agent. Please try again.`
    }
  }

  const generateAgentBasedResponse = async (agentType, agentResult) => {
    switch (agentType) {
      case 'summarize':
        return `**Summary:**\n\n${agentResult}`

      case 'highlight':
        if (Array.isArray(agentResult) && agentResult.length > 0) {
          const highlights = agentResult.slice(0, 5).map((h, index) => 
            `${index + 1}. **${h.timestamp}**: ${h.text}`
          ).join('\n\n')
          return `✨ **Highlights (${agentResult.length} parts):**\n\n${highlights}${agentResult.length > 5 ? '\n\n*And many more highlights...*' : ''}`
        }
        return 'No highlights found in this video.'

      case 'violation':
        if (Array.isArray(agentResult) && agentResult.length > 0) {
          const violations = agentResult.map((v, index) => 
            `${index + 1}. **${v.violation}**: ${v.explanation}`
          ).join('\n\n')
          return `⚠️ **Detected ${agentResult.length} potential violations:**\n\n${violations}`
        }
        return '✅ Don\'t worry, no violations found.'

      default:
        return 'Analysis is complete.'
    }
  }

  // Thông báo khi agent đang chạy
  const getAgentRunningMessage = (agentType) => {
    const messages = {
      'summarize': '🔄 Summarizing...',
      'highlight': '🔄 Extracting highlights... This may take a few seconds.',
      'violation': '🔄 Checking for policy violations... Please wait.'
    }
    return messages[agentType] || '🔄 Processing your request...'
  }

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!inputMessage.trim() || isLoading) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage.trim(),
      timestamp: new Date().toLocaleTimeString(),
      intent: 'user'
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setIsLoading(true)

    try {
      const response = await handleUserMessage(userMessage.content)
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.content,
        timestamp: new Date().toLocaleTimeString(),
        intent: response.intent
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Error handling message:', error)
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error while processing your message. Please try again.',
        timestamp: new Date().toLocaleTimeString(),
        intent: 'error'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage(e)
    }
  }

  const suggestedQuestions = transcriptData ? [
    "Summarize the video content",
    "What are the highlights in the video?",
    "Are there any policy violations?",
    "What is this video about?",
    "Analyze the content in detail",
    "Where are the important timestamps?"
  ] : [
    "How to upload a video?",
    "What file formats are supported?",
    "How does AI analysis work?",
    "What can you do?"
  ]

  const handleSuggestedQuestion = (question) => {
    setInputMessage(question)
  }

  // Clear chat history
  const handleClearChat = async () => {
    try {
      await chatService.clearChatHistory()
      setMessages([{
        id: 1,
        type: 'bot',
        content: 'History has been cleared. Welcome back!',
        timestamp: new Date().toLocaleTimeString(),
        intent: 'system'
      }])
      chatService.resetSession()
    } catch (error) {
      console.error('Error clearing chat:', error)
    }
  }

  // Display intent icon
  const getIntentIcon = (intent) => {
    switch (intent) {
      case 'rag_question':
        return <Database size={12} className="intent-icon" title="RAG Query" />
      case 'agent_automation':
        return <Brain size={12} className="intent-icon" title="Agent Automation" />
      case 'general_chat':
        return <MessageCircle size={12} className="intent-icon" title="General Chat" />
      default:
        return null
    }
  }

  return (
    <div className="chatbot-container">
      <div className="chat-header">
        <h2>
          <MessageCircle size={24} />
          AI Assistant
          {vectorStoreStatus === 'adding' && <Loader2 size={16} className="spin ml-2" />}
        </h2>
        <p>Ask me anything about the video...</p>
        {transcriptData && (
          <div className="transcript-status">
            📄 {transcriptData.config?.type_of_source || 'Video'} is ready
            {isTranscriptInVectorStore && <span className="vector-status">• 🔍 Information</span>}
          </div>
        )}
        <button onClick={handleClearChat} className="clear-chat-btn">
          Clear History
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.type} ${message.intent || ''}`}>
            <div className="message-avatar">
              {message.type === 'user' ? (
                <User size={16} />
              ) : (
                <Bot size={16} />
              )}
            </div>
            <div className="message-content">
              <div className="message-bubble">
                {message.content.split('\n').map((line, index) => (
                  <div key={index}>
                    {line.includes('**') ? (
                      <span dangerouslySetInnerHTML={{
                        __html: line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      }} />
                    ) : (
                      line
                    )}
                  </div>
                ))}
              </div>
              <div className="message-meta">
                <span className="message-time">{message.timestamp}</span>
                {message.intent && message.type === 'user' && getIntentIcon(message.intent)}
              </div>
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-avatar">
              <Bot size={16} />
            </div>
            <div className="message-content">
              <div className="message-bubble">
                <Loader2 size={14} className="spin" />
                Analyzing...
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {messages.length <= 1 && (
        <div className="suggested-questions">
          <h4>Suggested Questions:</h4>
          <div className="question-buttons">
            {suggestedQuestions.map((question, index) => (
              <button
                key={index}
                className="question-button"
                onClick={() => handleSuggestedQuestion(question)}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      )}

      <form onSubmit={handleSendMessage} className="chat-input-container">
        <input
          type="text"
          className="chat-input"
          placeholder={transcriptData ? 
            "Ask me anything..." : 
            "Please process the video first..."
          }
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button
          type="submit"
          className="send-button"
          disabled={!inputMessage.trim() || isLoading}
        >
          {isLoading ? <Loader2 size={16} className="spin" /> : <Send size={16} />}
        </button>
      </form>
    </div>
  )
}

export default Chatbot
