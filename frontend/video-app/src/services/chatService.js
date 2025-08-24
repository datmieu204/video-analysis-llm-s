// Chat service để kết nối với backend API
const API_BASE_URL = 'http://localhost:8000/api'

class ChatService {
  constructor() {
    this.sessionId = this.generateSessionId()
  }

  generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9)
  }

  // Intent Classification
  classifyIntent(message) {
    const lowerMessage = message.toLowerCase()

    // 1. Agent automation keywords
    const agentKeywords = {
      summarize: ['summary', 'summarize', 'tóm tắt', 'tổng hợp', 'tóm lược', 'summerize'],
      highlight: ['highlight', 'highlights', 'điểm nổi bật', 'phần quan trọng', 'điểm chính'],
      violation: ['violation', 'violations', 'vi phạm', 'kiểm duyệt', 'chính sách']
    }

    for (const [agentType, keywords] of Object.entries(agentKeywords)) {
      if (keywords.some(keyword => lowerMessage.includes(keyword))) {
        return {
          type: 'agent_automation',
          agentType: agentType,
          confidence: 0.9
        }
      }
    }

    // 2. RAG-related questions
    const ragKeywords = [
      'nội dung', 'transcript', 'video nói về', 'trong video', 'phần này',
      'thời điểm', 'timestamp', 'phút thứ', 'giây thứ', 'đoạn', 'phần',
      'người nói', 'tác giả', 'diễn giả', 'chi tiết', 'giải thích thêm',
      'what does', 'what is mentioned', 'tell me about', 'explain',
      'ở đâu', 'khi nào', 'như thế nào', 'tại sao', 'phân tích',
      'video', 'miêu tả', 'mô tả', 'describe', 'nói về', 'về gì', 
      'có nói', 'có liên quan', 'đề cập', 'thảo luận', 'react', 'javascript',
      'python', 'programming', 'coding', 'tutorial', 'đang miêu tả'
    ]

    if (ragKeywords.some(keyword => lowerMessage.includes(keyword))) {
      return {
        type: 'rag_question',
        confidence: 0.8
      }
    }

    // 3. General chat
    return {
      type: 'general_chat',
      confidence: 0.7
    }
  }

  // Add transcript to vector store
  async addTranscriptToVectorStore(transcriptId) {
    try {
      console.log('Adding transcript to vector store:', transcriptId) // Debug log
      const response = await fetch(`${API_BASE_URL}/chat/add-transcript`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          transcript_id: transcriptId
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('Add transcript response:', result) // Debug log
      return result
    } catch (error) {
      console.error('Error adding transcript to vector store:', error)
      throw error
    }
  }

  // General chat
  async askQuestion(question, includeHistory = true) {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          question: question,
          include_history: includeHistory
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('Chat response:', result) // Debug log
      return result
    } catch (error) {
      console.error('Error asking question:', error)
      throw error
    }
  }

  // Chat with RAG
  async askWithRAG(question, includeHistory = true) {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/ask-rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          question: question,
          include_history: includeHistory
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('RAG response:', result) // Debug log
      return result
    } catch (error) {
      console.error('Error asking with RAG:', error)
      throw error
    }
  }

  // Clear chat history
  async clearChatHistory() {
    try {
      const response = await fetch(`${API_BASE_URL}/chat/clear/${this.sessionId}`, {
        method: 'DELETE'
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      return result
    } catch (error) {
      console.error('Error clearing chat history:', error)
      throw error
    }
  }

  // Reset session
  resetSession() {
    this.sessionId = this.generateSessionId()
  }
}

export default new ChatService()