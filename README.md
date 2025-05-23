# 🤖 Chatbot Hỏi Đáp Tài Liệu PDF (DeepSeek + LangChain)

Ứng dụng chatbot đơn giản giúp bạn tải file PDF và đặt câu hỏi trực tiếp bằng tiếng Việt. Bot sẽ đọc nội dung và trả lời dựa trên nội dung tài liệu, sử dụng API DeepSeek và thư viện LangChain.

## 🚀 Tính năng

- Đọc tài liệu PDF và trích xuất văn bản
- Chia nhỏ văn bản và tạo embeddings tiếng Việt
- Truy vấn ngữ nghĩa sử dụng FAISS + DeepSeek
- Trả lời bằng tiếng Việt, thân thiện và dễ hiểu

## 🛠 Cài đặt

### 1. Tạo môi trường ảo (khuyến khích)
```bash
python -m venv venv
source venv/bin/activate  # hoặc venv\Scripts\activate nếu dùng Windows

``` 

## 2. Cài các thư viện cần thiết
```bash
pip install -r requirements.txt
``` 
## 3. Tạo file .env để lưu API Key

DEEPSEEK_API_KEY=your_deepseek_api_key_here

## ▶️ Chạy ứng dụng
```bash
streamlit run app.py
``` 
## 🧠 Mô hình sử dụng
```bash
Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

LLM: deepseek-chat thông qua API
```
## 📁 Cấu trúc thư mục gợi ý
```bash
📦chatbot-pdf
 ┣ 📄app.py
 ┣ 📄requirements.txt
 ┣ 📄.env
 ┗ 📄README.md
``` 
## 📬 Liên hệ
```bash
Zalo: 03*******
Phát triển bởi Thái, sử dụng LangChain + DeepSeek API.
```
