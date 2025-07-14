import os
import tempfile
import asyncio
import streamlit as st
from langchain_core.documents import Document as LangChainDocument
from langchain_community.document_loaders import WebBaseLoader
from llama_parse import LlamaParse

class FileHandler:
    def __init__(self):
        self.llama_api_key = st.secrets["LLAMA_CLOUD_API_KEY"]
    
    async def parse_with_llamaparse(self, uploaded_files):
        """LlamaParse를 사용한 파일 파싱"""
        parser = LlamaParse(
            api_key=self.llama_api_key,
            result_type="markdown",
            verbose=True,
        )
        
        parsed_data = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
                tmp_file.write(file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                documents = await parser.aload_data(tmp_file_path)
                parsed_data.extend(documents)
            except Exception as e:
                st.error(f"LlamaParse 처리 중 오류 발생 ({file.name}): {e}")
            finally:
                os.remove(tmp_file_path)
        
        return parsed_data

    def get_documents_from_files(self, uploaded_files):
        """파일에서 문서 추출"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        llama_index_documents = loop.run_until_complete(
            self.parse_with_llamaparse(uploaded_files)
        )
        
        if llama_index_documents:
            langchain_documents = [
                LangChainDocument(
                    page_content=doc.text, 
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {"source": "unknown"}
                )
                for doc in llama_index_documents
            ]
            return langchain_documents
        return []

    def get_documents_from_url(self, url):
        """URL에서 문서 추출"""
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                st.warning("URL에서 콘텐츠를 추출하지 못했습니다.")
                return []
            
            st.success(f"URL에서 {len(documents)}개 문서를 성공적으로 로드했습니다.")
            return documents
            
        except Exception as e:
            st.error(f"URL 로딩 오류: {e}")
            return []
