# Video Analysis with Large Language Models

Dự án này là một ứng dụng web hoàn chỉnh cho phép người dùng phân tích sâu nội dung video. Bằng cách cung cấp một URL video (ví dụ: từ YouTube), hệ thống sẽ tự động gỡ băng (transcribe), thực hiện các phân tích dựa trên AI, và cho phép người dùng "trò chuyện" với nội dung video thông qua một chatbot thông minh.

## ✨ Tính năng chính

- **Xử lý Video từ URL**: Dễ dàng nhập URL video từ các nền tảng phổ biến để bắt đầu phân tích.
- **Gỡ băng tự động**: Tích hợp mô hình **OpenAI Whisper** để chuyển đổi lời thoại trong video thành văn bản với độ chính xác cao.
- **Phân tích đa tác vụ bằng AI**:
    - **Tóm tắt nội dung**: Tự động tạo bản tóm tắt ngắn gọn, súc tích về nội dung chính của video.
    - **Trích xuất điểm nhấn**: Xác định và liệt kê các câu nói, ý tưởng quan trọng hoặc đáng chú ý nhất.
    - **Phát hiện vi phạm chính sách**: Sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) để đối chiếu nội dung video với một bộ quy tắc (ví dụ: chính sách của YouTube) và cảnh báo các vi phạm tiềm ẩn.
- **Chatbot tương tác (Hỏi-đáp với RAG)**: Đặt câu hỏi bằng ngôn ngữ tự nhiên về nội dung video và nhận câu trả lời chính xác, được trích xuất trực tiếp từ ngữ cảnh của video.
- **Giao diện Web hiện đại**: Giao diện người dùng được xây dựng bằng **React**, đảm bảo trải nghiệm mượt mà và trực quan.

## 🎥 Demo

![Video Demo](https://raw.githubusercontent.com/datmieu204/video-analysis-llm-s/main/demo.gif)


## 🏛️ Kiến trúc hệ thống

```
│   requirements.txt
├───app
│   │   app.py
│   │   main.py
│   │   test.py
│   ├───agents
│   │   │   chatbot.py
│   │   │   highlighter.py
│   │   │   summarizer.py
│   │   │   violence_detector.py
│   ├───api
│   │   │   db.py
│   │   │   routes.py
│   │   │   schemas.py
│   │   │   services.py
│   ├───core
│   │   │   build_store.py
│   │   │   embeddings.py
│   │   │   langgraph_flow.py
│   │   │   multi_llms.py
│   │   │   init.py
│   ├───data
│   ├───prompts
│   │   │   highlight_prompt.py
│   │   │   summarize_prompt.py
│   │   │   violation_prompt.py
│   ├───saved_transcripts
│   ├───transcript
│   │   │   transcription.py
│   │   │   whisper.py
│   │   ├───sources
│   │   │   │   audio_base.py
│   │   │   │   filelocal.py
│   │   │   │   youtube.py
│   ├───utils
│   │   │   config.py
│   │   │   crawl_law_data.ipynb
│   │   │   ffmpeg_utils.py
│   │   │   formatting.py
│   │   │   nlp_utils.py
│   │   │   splitter.py
│   ├───vectorstore
│   │   │   chroma.sqlite3
```

Hệ thống được xây dựng theo kiến trúc Client-Server, bao gồm hai thành phần chính:

1.  **Backend (Python/FastAPI)**:
    - **Framework**: FastAPI.
    - **Lõi AI**: Sử dụng **LangChain** và **LangGraph** để xây dựng và điều phối các "tác nhân AI" (AI Agents).
    - **Mô hình ngôn ngữ**: Tận dụng sức mạnh của các mô hình từ OpenAI (ví dụ: `gpt-4o-mini`).
    - **Knowledge Base**: Sử dụng **ChromaDB** làm cơ sở dữ liệu vector (Vector Database) để lưu trữ embeddings và hỗ trợ kỹ thuật RAG.
    - **Xử lý đa phương tiện**: Dùng `yt-dlp` để tải video và `ffmpeg` để xử lý âm thanh.

2.  **Frontend (React/Vite)**:
    - **Framework**: React.
    - **Build Tool**: Vite.
    - **Giao diện**: Các component được thiết kế để hiển thị kết quả phân tích và tương tác với chatbot một cách hiệu quả.

### Luồng hoạt động

1.  **Người dùng** cung cấp URL video qua giao diện Frontend.
2.  **Frontend** gửi yêu cầu đến API `/process_video` của **Backend**.
3.  **Backend** tải âm thanh, dùng **Whisper** để gỡ băng.
4.  Văn bản transcript được chia nhỏ, chuyển thành vector (embeddings) và lưu vào **ChromaDB**.
5.  Một luồng **LangGraph** được kích hoạt, điều phối các Agent (Summarizer, Highlighter, ViolenceDetector) thực hiện phân tích.
6.  Kết quả phân tích được trả về cho **Frontend** và hiển thị.
7.  Khi người dùng chat, **Frontend** gọi API `/chat`. **Backend** sử dụng RAG để truy vấn thông tin từ **ChromaDB** và tạo câu trả lời.

## ⚙️ Hướng dẫn cài đặt và chạy dự án

### Yêu cầu môi trường
- **Python** 3.9+
- **Node.js** 18+
- **FFmpeg**: Phải được cài đặt và thêm vào biến môi trường `PATH` của hệ thống.

### 1. Cài đặt Backend

```bash
# 1. Di chuyển vào thư mục backend
cd backend

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
# Trên Windows:
.\venv\Scripts\activate
# Trên macOS/Linux:
# source venv/bin/activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Tạo file .env và cung cấp API key
# Tạo một file tên là .env trong thư mục backend và thêm nội dung sau:
# OPENAI_API_KEY="your_openai_api_key_here"

# 5. Chạy server backend
uvicorn app.app:app --reload --port 8000
```

### 2. Cài đặt Frontend

```bash
# 1. Mở một terminal khác và di chuyển vào thư mục frontend
cd frontend/video-app

# 2. Cài đặt các gói phụ thuộc
npm install

# 3. Chạy ứng dụng frontend
npm run dev
```

### 3. Truy cập ứng dụng
Sau khi cả hai server đã chạy, mở trình duyệt và truy cập: `http://localhost:5173`

## 🚀 Hướng phát triển
- **Phân tích đa phương thức (Multimodal)**: Phân tích cả hình ảnh trong video để phát hiện hành động, đối tượng.
- **Tối ưu hóa hiệu năng**: Cải thiện tốc độ gỡ băng và phân tích.
- **Mở rộng bộ Agent**: Thêm các agent mới như phân tích cảm xúc, nhận diện thực thể.
- **Triển khai lên Cloud**: Đóng gói ứng dụng bằng Docker và triển khai lên các nền tảng đám mây.
