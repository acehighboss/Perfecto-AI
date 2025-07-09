# rag_pipeline.py

import streamlit as st
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PlaywrightURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from file_handler import get_documents_from_files

def get_retriever_from_source(source_type, source_input):
    documents = [] 
    with st.status("문서 처리 중...", expanded=True) as status:
        if source_type == "URL":
            status.update(label="URL 컨텐츠를 로드 중입니다...")
            loader = PlaywrightURLLoader(urls=[source_input], remove_selectors=["header", "footer"])
            documents = asyncio.run(loader.aload())
        elif source_type == "Files":
            status.update(label="파일을 파싱하고 있습니다...")
            documents = get_documents_from_files(source_input)

        if not documents:
            status.update(label="문서 로딩 실패.", state="error")
            return None

        status.update(label="문서를 청크(chunk)로 분할 중입니다...")
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n", chunk_size=500, chunk_overlap=50,
        )
        splits = text_splitter.split_documents(documents)
        
        status.update(label=f"임베딩 모델을 로컬에 로드 중입니다...")
        embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        
        status.update(label=f"{len(splits)}개의 청크를 임베딩하고 있습니다...")
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        status.update(label="Retriever를 생성 중입니다...")
        llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0, request_timeout=120)
        
        base_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        status.update(label="문서 처리 완료!", state="complete")
    
    return compression_retriever

# '답변 생성 체인'만 만들도록 역할을 명확히 합니다.
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
