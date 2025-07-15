import os
import pickle
from typing import List, Dict, Any, Optional
import streamlit as st

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain_core.retrievers import BaseRetriever

class RAGPipeline:
    def __init__(self, google_api_key: str):
        # Google AI 모델 초기화
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-004",
            google_api_key=google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            convert_system_message_to_human=True
        )
        
        # 벡터스토어 초기화
        self.vectorstore = None
        self.retriever = None
        
        # 저장된 벡터스토어 로드 시도
        self.load_vectorstore()
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """문서들로부터 벡터스토어 생성"""
        try:
            if not documents:
                st.error("처리할 문서가 없습니다.")
                return False
            
            # 진행률 표시
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("벡터스토어 생성 중...")
            
            if self.vectorstore is None:
                # 새 벡터스토어 생성
                self.vectorstore = FAISS.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                # 기존 벡터스토어에 문서 추가
                self.vectorstore.add_documents(documents)
            
            progress_bar.progress(0.8)
            
            # 리트리버 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            progress_bar.progress(1.0)
            
            # 벡터스토어 저장
            self.save_vectorstore()
            
            # 진행률 표시 제거
            progress_bar.empty()
            status_text.empty()
            
            return True
            
        except Exception as e:
            st.error(f"벡터스토어 생성 중 오류: {str(e)}")
            return False
    
    def search_similar_documents(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서 검색"""
        try:
            if self.retriever is None:
                return []
            
            # 문서 검색
            docs = self.retriever.get_relevant_documents(query)
            
            # 결과 포맷팅
            results = []
            for doc in docs[:k]:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'Unknown'),
                    'metadata': doc.metadata
                })
            
            return results
            
        except Exception as e:
            st.error(f"문서 검색 중 오류: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]], system_prompt: str = "") -> str:
        """컨텍스트를 바탕으로 답변 생성"""
        try:
            # 컨텍스트 구성
            context = ""
            for i, doc in enumerate(context_docs, 1):
                source = doc.get('source', 'Unknown')
                content = doc.get('content', '')
                context += f"\n[출처 {i}] {source}\n{content}\n"
            
            # 프롬프트 템플릿 생성
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", f"""당신은 업로드된 문서들을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

{system_prompt}

답변 시 다음 규칙을 따라주세요:
1. 제공된 컨텍스트만을 바탕으로 답변하세요
2. 답변 끝에 참고한 출처를 명시하세요
3. 컨텍스트에서 답을 찾을 수 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
4. 한국어로 자연스럽고 정확하게 답변하세요
5. 구체적이고 상세한 답변을 제공하세요"""),
                ("human", f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

질문: {query}

답변:""")
            ])
            
            # 체인 생성 및 실행
            chain = prompt_template | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            return response
            
        except Exception as e:
            st.error(f"답변 생성 중 오류: {str(e)}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def create_rag_chain(self, system_prompt: str = ""):
        """RAG 체인 생성"""
        if self.retriever is None:
            return None
        
        # 프롬프트 템플릿
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""당신은 업로드된 문서들을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

{system_prompt}

답변 시 다음 규칙을 따라주세요:
1. 제공된 컨텍스트만을 바탕으로 답변하세요
2. 답변 끝에 참고한 출처를 명시하세요
3. 컨텍스트에서 답을 찾을 수 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
4. 한국어로 자연스럽고 정확하게 답변하세요"""),
            ("human", """컨텍스트: {context}

질문: {question}

답변:""")
        ])
        
        # RAG 체인 구성
        def format_docs(docs):
            formatted = ""
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown')
                formatted += f"\n[출처 {i}] {source}\n{doc.page_content}\n"
            return formatted
        
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def save_vectorstore(self):
        """벡터스토어 저장"""
        try:
            if self.vectorstore is not None:
                self.vectorstore.save_local("faiss_vectorstore")
        except Exception as e:
            st.error(f"벡터스토어 저장 중 오류: {str(e)}")
    
    def load_vectorstore(self):
        """저장된 벡터스토어 로드"""
        try:
            if os.path.exists("faiss_vectorstore"):
                self.vectorstore = FAISS.load_local(
                    "faiss_vectorstore", 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                self.retriever = self.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
        except Exception as e:
            st.warning(f"기존 벡터스토어 로드 실패, 새로 생성합니다: {str(e)}")
            self.vectorstore = None
            self.retriever = None
    
    def clear_vectorstore(self):
        """벡터스토어 초기화"""
        try:
            # 파일 삭제
            import shutil
            if os.path.exists("faiss_vectorstore"):
                shutil.rmtree("faiss_vectorstore")
            
            # 객체 초기화
            self.vectorstore = None
            self.retriever = None
            
            return True
        except Exception as e:
            st.error(f"벡터스토어 초기화 중 오류: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        try:
            if self.vectorstore is not None:
                return self.vectorstore.index.ntotal
            return 0
        except:
            return 0
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """벡터스토어 정보 반환"""
        if self.vectorstore is None:
            return {"document_count": 0, "index_size": 0}
        
        try:
            return {
                "document_count": self.vectorstore.index.ntotal,
                "index_size": self.vectorstore.index.d
            }
        except:
            return {"document_count": 0, "index_size": 0}
