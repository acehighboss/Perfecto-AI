# rag_pipeline.py

import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
# [수정 1] HuggingFaceEmbeddings를 import 합니다.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

        status.update(label="문서를 청크(chunk)로 분할 중입니다...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
            is_separator_regex=False,
        )
        splits = text_splitter.split_documents(documents)
        
        # [수정 2] 임베딩 모델을 HuggingFaceEmbeddings와 bge-m3로 교체하고, 설정을 최적화합니다.
        status.update(label=f"임베딩 모델(BAAI/bge-m3)을 로컬에 로드 중입니다. 시간이 매우 오래 걸릴 수 있습니다...")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        status.update(label=f"{len(splits)}개의 청크를 임베딩하고 있습니다...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 3, 'fetch_k': 20}
        )
        status.update(label="문서 처리 완료!", state="complete")
    
    return retriever

def get_document_chain(system_prompt):
    template = f"""{system_prompt}

Answer the user's question based on the context provided below and the conversation history.
The context may include text and tables in markdown format. You must be able to understand and answer based on them.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Context:
{{context}}
"""
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return document_chain

def get_default_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)
    return prompt | llm | StrOutputParser()
