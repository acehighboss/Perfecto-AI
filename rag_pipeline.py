import google.generativeai as genai
import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
import streamlit as st

class RAGPipeline:
    def __init__(self, google_api_key: str):
        # Google AI 설정
        genai.configure(api_key=google_api_key)
        self.embedding_model = genai.GenerativeModel('models/embedding-004')
        self.llm_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # FAISS 설정
        self.dimension = 768  # embedding-004의 차원
        self.index = None
        self.documents = []  # 문서 텍스트 저장
        self.metadatas = []  # 메타데이터 저장
        
        # 저장된 인덱스 로드 시도
        self.load_index()
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트의 임베딩 벡터 생성"""
        try:
            result = genai.embed_content(
                model="models/embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(result['embedding'], dtype=np.float32)
        except Exception as e:
            st.error(f"임베딩 생성 중 오류: {str(e)}")
            return None
    
    def initialize_index(self):
        """FAISS 인덱스 초기화"""
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (코사인 유사도)
        self.documents = []
        self.metadatas = []
    
    def add_documents(self, chunks: List[str], source: str) -> bool:
        """문서 청크들을 FAISS 인덱스에 추가"""
        try:
            if self.index is None:
                self.initialize_index()
            
            embeddings = []
            valid_chunks = []
            valid_metadatas = []
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    embedding = self.get_embedding(chunk)
                    if embedding is not None:
                        # L2 정규화 (코사인 유사도를 위해)
                        embedding = embedding / np.linalg.norm(embedding)
                        embeddings.append(embedding)
                        valid_chunks.append(chunk)
                        valid_metadatas.append({
                            "source": source, 
                            "chunk_id": len(self.documents) + len(valid_chunks) - 1
                        })
            
            if embeddings:
                # FAISS 인덱스에 추가
                embeddings_array = np.vstack(embeddings)
                self.index.add(embeddings_array)
                
                # 문서와 메타데이터 저장
                self.documents.extend(valid_chunks)
                self.metadatas.extend(valid_metadatas)
                
                # 인덱스 저장
                self.save_index()
                return True
            
            return False
            
        except Exception as e:
            st.error(f"문서 추가 중 오류: {str(e)}")
            return False
    
    def search_similar_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서 검색"""
        try:
            if self.index is None or self.index.ntotal == 0:
                return []
            
            query_embedding = self.get_embedding(query)
            if query_embedding is None:
                return []
            
            # L2 정규화
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            query_embedding = query_embedding.reshape(1, -1)
            
            # FAISS 검색
            k = min(n_results, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            similar_docs = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    similar_docs.append({
                        'content': self.documents[idx],
                        'source': self.metadatas[idx]['source'],
                        'similarity': float(score)  # 코사인 유사도 점수
                    })
            
            return similar_docs
            
        except Exception as e:
            st.error(f"문서 검색 중 오류: {str(e)}")
            return []
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]], system_prompt: str = "") -> str:
        """컨텍스트를 바탕으로 답변 생성"""
        try:
            # 컨텍스트 구성
            context = ""
            sources = []
            
            for i, doc in enumerate(context_docs, 1):
                context += f"\n[출처 {i}] {doc['source']}\n{doc['content']}\n"
                sources.append(doc['source'])
            
            # 프롬프트 구성
            base_prompt = f"""
당신은 업로드된 문서들을 바탕으로 질문에 답변하는 AI 어시스턴트입니다.

{system_prompt}

다음 컨텍스트를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

질문: {query}

답변 시 다음 규칙을 따라주세요:
1. 제공된 컨텍스트만을 바탕으로 답변하세요
2. 답변 끝에 참고한 출처를 명시하세요
3. 컨텍스트에서 답을 찾을 수 없다면 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 답변하세요
4. 한국어로 자연스럽고 정확하게 답변하세요

답변:
"""
            
            response = self.llm_model.generate_content(base_prompt)
            return response.text
            
        except Exception as e:
            st.error(f"답변 생성 중 오류: {str(e)}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def save_index(self):
        """FAISS 인덱스와 메타데이터 저장"""
        try:
            if self.index is not None:
                # FAISS 인덱스 저장
                faiss.write_index(self.index, "faiss_index.bin")
                
                # 문서와 메타데이터 저장
                with open("documents_metadata.pkl", "wb") as f:
                    pickle.dump({
                        'documents': self.documents,
                        'metadatas': self.metadatas
                    }, f)
        except Exception as e:
            st.error(f"인덱스 저장 중 오류: {str(e)}")
    
    def load_index(self):
        """저장된 FAISS 인덱스와 메타데이터 로드"""
        try:
            if os.path.exists("faiss_index.bin") and os.path.exists("documents_metadata.pkl"):
                # FAISS 인덱스 로드
                self.index = faiss.read_index("faiss_index.bin")
                
                # 문서와 메타데이터 로드
                with open("documents_metadata.pkl", "rb") as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadatas = data['metadatas']
            else:
                self.initialize_index()
        except Exception as e:
            st.error(f"인덱스 로드 중 오류: {str(e)}")
            self.initialize_index()
    
    def clear_database(self):
        """데이터베이스 초기화"""
        try:
            # 파일 삭제
            if os.path.exists("faiss_index.bin"):
                os.remove("faiss_index.bin")
            if os.path.exists("documents_metadata.pkl"):
                os.remove("documents_metadata.pkl")
            
            # 인덱스 초기화
            self.initialize_index()
            return True
        except Exception as e:
            st.error(f"데이터베이스 초기화 중 오류: {str(e)}")
            return False
    
    def get_document_count(self) -> int:
        """저장된 문서 수 반환"""
        try:
            if self.index is not None:
                return self.index.ntotal
            return 0
        except:
            return 0
