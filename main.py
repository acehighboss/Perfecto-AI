import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline

# 페이지 설정
st.set_page_config(page_title="Multimodal RAG Chatbot", page_icon="🤖")
st.title("🤖 멀티모달 파일/URL 분석 RAG 챗봇")
st.markdown(
    """
    안녕하세요! 이 챗봇은 웹사이트 URL이나 업로드된 파일(PDF, DOCX)의 내용을 분석하고 답변합니다.
    **LlamaParse**를 사용하여 **테이블과 텍스트를 함께 인식**하고 질문에 답할 수 있습니다.
    """
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 문서 분석 전문가 AI 어시스턴트입니다. 주어진 문서의 텍스트와 테이블을 정확히 이해하고 상세하게 답변해주세요."

# 핸들러 및 파이프라인 초기화
@st.cache_resource
def initialize_components():
    file_handler = FileHandler()
    rag_pipeline = RAGPipeline()
    return file_handler, rag_pipeline

file_handler, rag_pipeline = initialize_components()

def process_source(source_type, source_input):
    """소스 처리 및 검색기 생성"""
    documents = []
    
    if source_type == "URL":
        documents = file_handler.get_documents_from_url(source_input)
    elif source_type == "Files":
        documents = file_handler.get_documents_from_files(source_input)
    
    if documents:
        return rag_pipeline.create_retriever(documents)
    return None

def display_sources(source_documents):
    """출처 표시"""
    if source_documents:
        with st.expander("참고한 출처 보기 (마크다운 형식)"):
            for i, source in enumerate(source_documents):
                st.text(f"--- 출처 {i+1} ---")
                st.markdown(source.page_content)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    st.divider()
    
    # AI 페르소나 설정
    st.subheader("🤖 AI 페르소나 설정")
    system_prompt_input = st.text_area(
        "AI의 역할을 설정해주세요.", 
        value=st.session_state.system_prompt, 
        height=150
    )
    st.session_state.system_prompt = system_prompt_input
    
    st.divider()
    
    # 분석 대상 설정
    st.subheader("🔎 분석 대상 설정")
    
    # URL 입력
    url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "파일 업로드 (PDF, DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    
    st.info("LlamaParse는 테이블, 텍스트가 포함된 문서 분석에 최적화되어 있습니다.", icon="ℹ️")
    
    # 분석 시작 버튼
    if st.button("분석 시작", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        
        if uploaded_files:
            with st.spinner("LlamaParse로 문서를 분석하고 있습니다..."):
                st.session_state.retriever = process_source("Files", uploaded_files)
        elif url_input:
            with st.spinner("URL을 분석하고 있습니다..."):
                st.session_state.retriever = process_source("URL", url_input)
        else:
            st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

        if st.session_state.retriever:
            st.success("분석이 완료되었습니다! 이제 질문해보세요.")
    
    st.divider()
    
    # 대화 초기화 버튼
    if st.button("대화 초기화", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# 메인 채팅 인터페이스
# 이전 메시지 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

# 사용자 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    try:
        # 채팅 히스토리 생성
        chat_history = rag_pipeline.format_chat_history(st.session_state.messages)
        
        if st.session_state.retriever:
            # RAG 체인 사용
            chain = rag_pipeline.create_conversational_rag_chain(
                st.session_state.retriever, 
                st.session_state.system_prompt
            )
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                source_documents = []
                
                # 스트리밍 응답
                for chunk in chain.stream({
                    "input": user_input, 
                    "chat_history": chat_history
                }):
                    if "answer" in chunk:
                        ai_answer += chunk["answer"]
                        container.markdown(ai_answer)
                    if "context" in chunk and not source_documents:
                        source_documents = chunk["context"]
                
                # 메시지 저장
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": source_documents
                })
                
                # 출처 표시
                display_sources(source_documents)
        else:
            # 기본 체인 사용
            chain = rag_pipeline.create_default_chain(st.session_state.system_prompt)
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                
                for token in chain.stream({
                    "question": user_input, 
                    "chat_history": chat_history
                }):
                    ai_answer += token
                    container.markdown(ai_answer)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": []
                })
    
    except Exception as e:
        st.chat_message("assistant").error(f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()
