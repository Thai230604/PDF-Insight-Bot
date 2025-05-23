import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Tải biến môi trường
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

st.header("👾Chatbot Hỏi Đáp Tài Liệu PDF")

with st.sidebar:
    st.title("Tải Tài Liệu")
    file = st.file_uploader("Tải lên file PDF để bắt đầu hỏi đáp", type="pdf")

# Định nghĩa prompt tối ưu cho tiếng Việt
prompt_template = """
Bạn là một trợ lý AI thông minh, am hiểu văn hóa và ngôn ngữ Việt Nam. Hãy trả lời câu hỏi của người dùng dựa trên nội dung tài liệu được cung cấp. 
Tất cả câu trả lời phải được viết bằng tiếng Việt, sử dụng ngôn ngữ tự nhiên, dễ hiểu, và phù hợp với ngữ cảnh Việt Nam. 
Nếu không tìm thấy thông tin liên quan trong tài liệu, hãy nói rõ rằng thông tin không có và cung cấp câu trả lời hợp lý dựa trên kiến thức chung.

Câu hỏi: {question}
Nội dung tài liệu: {context}
Trả lời:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

if file is not None:
    # Đọc nội dung PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Xử lý trang không có văn bản

    # Chia văn bản thành các đoạn
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    st.write(chunks)  # Bỏ comment nếu muốn hiển thị các đoạn văn bản

    # Sử dụng mô hình nhúng tiếng Việt
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


    # Tạo vector store với FAISS
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Khởi tạo DeepSeek API client
    llm = ChatOpenAI(
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base="https://api.deepseek.com/v1",  # base_url của DeepSeek
        model_name="deepseek-chat",  # hoặc tên mô hình cụ thể của DeepSeek
        temperature=0.7
)

    # Tạo chuỗi truy vấn với DeepSeek API
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
)

    # Nhập câu hỏi từ người dùng
    user_question = st.text_input("Nhập câu hỏi của bạn tại đây")

    if user_question:
        # Thêm hướng dẫn trả lời bằng tiếng Việt
        user_question = f"Hãy trả lời bằng tiếng Việt: {user_question}"
        # Thực hiện truy vấn
        response = chain({"question": user_question, "chat_history": []})
        st.write(response["answer"])  # Hiển thị câu trả lời