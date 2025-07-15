import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import re
import tiktoken

class IntelligentTextSplitter:
    """문단별/문장별 지능형 분할 시스템"""
    
    def __init__(self, max_tokens=3000):
        self.max_tokens = max_tokens
        self.encoder = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text):
        """토큰 수 계산"""
        return len(self.encoder.encode(text))
    
    def split_by_sentences(self, text):
        """문장별 분할"""
        sentence_patterns = [
            r'[.!?]\s+',
            r'[。！？]\s*',
            r'\n\s*\n',
        ]
        
        sentences = []
        current_text = text
        
        for pattern in sentence_patterns:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                sentences.extend([part.strip() for part in parts if part.strip()])
                break
        
        if not sentences:
            sentences = [text]
        
        return sentences
    
    def split_by_paragraphs(self, text):
        """문단별 분할"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def smart_split(self, text):
        """지능형 분할 (문단 우선, 필요시 문장별)"""
        paragraphs = self.split_by_paragraphs(text)
        chunks = []
        
        for paragraph in paragraphs:
            token_count = self.count_tokens(paragraph)
            
            if token_count <= self.max_tokens:
                chunks.append(paragraph)
            else:
                # 문단이 너무 길면 문장별로 분할
                sentences = self.split_by_sentences(paragraph)
                current_chunk = ""
                current_tokens = 0
                
                for sentence in sentences:
                    sentence_tokens = self.count_tokens(sentence)
                    
                    if sentence_tokens > self.max_tokens:
                        # 문장도 너무 길면 강제 분할
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                            current_tokens = 0
                        
                        # 긴 문장을 토큰 단위로 분할
                        tokens = self.encoder.encode(sentence)
                        for i in range(0, len(tokens), self.max_tokens):
                            chunk_tokens = tokens[i:i+self.max_tokens]
                            chunk_text = self.encoder.decode(chunk_tokens)
                            chunks.append(chunk_text)
                    else:
                        if current_tokens + sentence_tokens > self.max_tokens:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            if current_chunk:
                                current_chunk += " " + sentence
                            else:
                                current_chunk = sentence
                            current_tokens += sentence_tokens
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        return chunks
    
    def split_documents(self, documents):
        """문서 분할"""
        split_docs = []
        
        for doc in documents:
            chunks = self.smart_split(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    split_docs.append(Document(
                        page_content=chunk,
                        metadata={
                            **doc.metadata,
                            'chunk_id': i,
                            'token_count': self.count_tokens(chunk)
                        }
                    ))
        
        return split_docs

class RAGPipeline:
    def __init__(self):
        self.google_api_key = st.secrets["GOOGLE_API_KEY"]
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.google_api_key
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=0,
            google_api_key=self.google_api_key
        )
        
        self.text_splitter = IntelligentTextSplitter(max_tokens=3000)

    def get_system_prompt_template(self, system_prompt):
        """시스템 프롬프트 템플릿"""
        template = system_prompt + """

다음 컨텍스트를 바탕으로 사용자의 질문에 정확하게 답변해주세요.

**중요한 지침:**
1. 제공된 컨텍스트를 꼼꼼히 검토하고 관련 정보를 찾아 답변하세요
2. 컨텍스트에 관련 내용이 있다면 반드시 활용하여 답변하세요
3. 답변할 때는 참조한 출처를 명시해주세요
4. 정확하고 구체적인 답변을 제공하세요
5. 컨텍스트를 충분히 활용하여 도움이 되는 답변을 하세요

컨텍스트:
{context}
"""
        return template

    def create_retriever(self, documents):
        """향상된 검색기 생성"""
        if not documents:
            st.warning("문서에서 내용을 추출하지 못했습니다.")
            return None

        # 지능형 분할 적용
        splits = self.text_splitter.split_documents(documents)
        
        if not splits:
            st.warning("문서 분할에 실패했습니다.")
            return None
        
        # 토큰 수 정보 표시
        total_tokens = sum(doc.metadata.get('token_count', 0) for doc in splits)
        st.info(f"📊 지능형 분할 완료: {len(splits)}개 청크, 총 {total_tokens:,} 토큰")
        
        # FAISS 벡터스토어 생성
        try:
            vectorstore = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            st.error(f"벡터스토어 생성 오류: {e}")
            return None

        # 더 많은 문서를 검색하여 관련 내용 누락 방지
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 5,  # 더 많은 문서 검색
                "score_threshold": 0.5  # 낮은 임계값으로 더 많은 후보 포함
            }
        )
        
        return retriever

    def create_conversational_rag_chain(self, retriever, system_prompt):
        """대화형 RAG 체인 생성"""
        enhanced_template = self.get_system_prompt_template(system_prompt)
        
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", enhanced_template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ])
        
        document_chain = create_stuff_documents_chain(self.llm, rag_prompt)
        return create_retrieval_chain(retriever, document_chain)

    def create_default_chain(self, system_prompt):
        """기본 체인 생성"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ])
        return prompt | self.llm | StrOutputParser()

    def format_chat_history(self, messages):
        """채팅 히스토리 포맷팅"""
        chat_history = []
        for msg in messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            else:
                chat_history.append(AIMessage(content=msg["content"]))
        return chat_history
