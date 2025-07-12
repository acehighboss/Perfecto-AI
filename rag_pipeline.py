# rag_pipeline.py (Markdown 분할 방식 적용)

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import SeleniumURLLoader
# [수정 1] MarkdownHeaderTextSplitter와 RecursiveCharacterTextSplitter를 import
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    documents = [] 
    with st.status("문서 처리 중...", expanded=True) as status:
        if source_type == "URL":
            status.update(label="URL 컨텐츠를 로드 중입니다...")
            loader = SeleniumURLLoader(urls=[source_input])
            documents = loader.load()
        elif source_type == "Files":
            status.update(label="파일을 파싱하고 있습니다...")
            documents = get_documents_from_files(source_input)

        if not documents:
            status.update(label="문서 로딩 실패.", state="error")
            return None

        status.update(label="문서를 구조적으로 분할 중입니다...")
        # [수정 2] 계층적 분할 적용
        # 1. Markdown 제목 기준으로 1차 분할
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, return_each_line=False)
        
        all_splits = []
        for doc in documents:
            md_header_splits = markdown_splitter.split_text(doc.page_content)
            all_splits.extend(md_header_splits)

        # 2. 1차 분할된 청크가 너무 길 경우를 대비해, 글자 수 기준으로 2차 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        splits = text_splitter.split_documents(all_splits)
        
        status.update(label=f"임베딩 모델을 로컬에 로드 중입니다...")
        embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        
        status.update(label=f"{len(splits)}개의 청크를 임베딩하고 있습니다...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 5, 'fetch_k': 25} # 후보군을 조금 더 늘림
        )
        status.update(label="문서 처리 완료!", state="complete")
    
    return retriever
