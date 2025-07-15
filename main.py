import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline

# 페이지 설정
st.set_page_config(page_title="Universal RAG Chatbot", page_icon="🤖")
st.title("🤖 범용 RAG 챗봇")
st.markdown(
    """
    모든 종류의 문서와 질문에 대응하는 범용 RAG 챗봇입니다.
    **PDF, DOCX, TXT** 파일과 **URL**을 분석하여 정확한 답변을 제공합니다.
    """
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "당신은 범용 문서 분석 전문가 AI 어시스턴트입니다. 모든 종류의 문서와 주제에 대해 정확하고 상세한 답변을 제공합니다."

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
        total_length = sum(len(doc.page_content) for doc in documents)
        st.info(f"📄 추출된 문서: {len(documents)}개, 총 {total_length:,}자")
        
        # 첫 번째 문서의 일부 내용 표시
        if documents[0].page_content:
            preview = documents[0].page_content[:300] + "..." if len(documents[0].page_content) > 300 else documents[0].page_content
            with st.expander("📋 추출된 내용 미리보기"):
                st.text(preview)
        
        return rag_pipeline.create_retriever(documents)
    else:
        st.error("문서를 추출하지 못했습니다.")
        return None

def display_sources(source_documents):
    """출처 표시"""
    if source_documents:
        with st.expander("📚 참고 출처 보기"):
            for i, source in enumerate(source_documents):
                st.text(f"--- 출처 {i+1} ---")
                if hasattr(source, 'metadata') and source.metadata:
                    file_source = source.metadata.get("source", "unknown")
                    st.caption(f"📁 {file_source}")
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
        height=150,
        key="system_prompt_input"
    )
    
    if st.button("🎯 페르소나 적용", type="primary", use_container_width=True):
        st.session_state.system_prompt = system_prompt_input
        st.success("✅ AI 페르소나가 적용되었습니다!")
    
    st.divider()
    
    # 분석 대상 설정
    st.subheader("🔎 분석 대상 설정")
    
    # URL 입력
    url_input = st.text_input("웹사이트 URL", placeholder="https://example.com")
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "파일 업로드", 
        type=["pdf", "docx", "txt"], 
        accept_multiple_files=True,
        help="PDF, DOCX, TXT 파일을 업로드할 수 있습니다"
    )
    
    # 분석 시작 버튼
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        
        if uploaded_files:
            with st.spinner("파일을 분석하고 있습니다..."):
                st.session_state.retriever = process_source("Files", uploaded_files)
        elif url_input:
            with st.spinner("URL을 분석하고 있습니다..."):
                st.session_state.retriever = process_source("URL", url_input)
        else:
            st.warning("분석할 URL을 입력하거나 파일을 업로드해주세요.")

        if st.session_state.retriever:
            st.success("✅ 분석이 완료되었습니다! 이제 질문해보세요.")
        else:
            st.error("❌ 분석에 실패했습니다. 다시 시도해주세요.")
    
    st.divider()
    
    # 현재 적용된 페르소나 표시
    st.subheader("📋 현재 적용된 페르소나")
    with st.expander("현재 페르소나 보기"):
        st.text(st.session_state.system_prompt)
    
    st.divider()
    
    # 대화 초기화 버튼
    if st.button("🔄 대화 초기화", type="secondary", use_container_width=True):
        system_prompt_backup = st.session_state.get("system_prompt", "")
        st.session_state.clear()
        st.session_state["messages"] = []
        st.session_state["system_prompt"] = system_prompt_backup
        st.session_state.retriever = None
        st.success("🔄 대화가 초기화되었습니다! (페르소나는 유지됨)")
        st.rerun()

# 메인 채팅 인터페이스
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

# 사용자 입력
user_input = st.chat_input("어떤 질문이든 자유롭게 물어보세요!")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    
    try:
        chat_history = rag_pipeline.format_chat_history(st.session_state.messages)
        
        if st.session_state.retriever:
            chain = rag_pipeline.create_conversational_rag_chain(
                st.session_state.retriever, 
                st.session_state.system_prompt
            )
            
            with st.chat_message("assistant"):
                container = st.empty()
                ai_answer = ""
                source_documents = []
                
                for chunk in chain.stream({
                    "input": user_input, 
                    "chat_history": chat_history
                }):
                    if "answer" in chunk:
                        ai_answer += chunk["answer"]
                        container.markdown(ai_answer)
                    if "context" in chunk and not source_documents:
                        source_documents = chunk["context"]
                
                if source_documents:
                    st.info(f"🔍 {len(source_documents)}개의 관련 문서를 찾았습니다.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": source_documents
                })
                
                display_sources(source_documents)
        else:
            st.warning("⚠️ 분석된 문서가 없어 일반 모드로 답변합니다.")
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
        st.chat_message("assistant").error(f"답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()
