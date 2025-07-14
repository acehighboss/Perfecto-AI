import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline
import time

# 페이지 설정
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 RAG 챗봇")
st.markdown(
    """
    **정확한 출처 기반 답변을 제공하는 RAG 챗봇입니다.**
    문서나 URL을 업로드하고 관련 질문을 하면 출처와 함께 정확한 답변을 받을 수 있습니다.
    """
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """당신은 문서 분석 전문가 AI 어시스턴트입니다. 
제공된 문서의 내용을 정확히 이해하고 사용자의 질문에 대해 출처를 명시하며 정확한 답변을 제공합니다.
추측이나 가정 없이 오직 문서에 기반한 정보만을 제공합니다."""

# 핸들러 및 파이프라인 초기화
@st.cache_resource
def initialize_components():
    file_handler = FileHandler()
    rag_pipeline = RAGPipeline()
    return file_handler, rag_pipeline

file_handler, rag_pipeline = initialize_components()

def process_source(source_type, source_input):
    """소스 처리 및 검색기 생성 (진행 상황 표시)"""
    documents = []
    
    if source_type == "URL":
        documents = file_handler.get_documents_from_url(source_input)
    elif source_type == "Files":
        # 파일 크기 확인
        total_size = sum(file.size for file in source_input)
        size_mb = total_size / (1024 * 1024)
        
        if size_mb > 5:
            st.info(f"파일 크기: {size_mb:.1f}MB - 처리 시간이 다소 걸릴 수 있습니다.")
        
        documents = file_handler.get_documents_from_files(source_input)
    
    if documents:
        return rag_pipeline.create_retriever(documents)
    return None

def display_sources(source_documents):
    """출처 표시"""
    if source_documents:
        with st.expander("📚 참고 출처 보기"):
            for i, source in enumerate(source_documents):
                st.text(f"--- 출처 {i+1} ---")
                st.markdown(source.page_content)
                if hasattr(source, 'metadata') and source.metadata:
                    st.json(source.metadata)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 시스템 프롬프트 설정
    st.subheader("🤖 시스템 프롬프트 설정")
    system_prompt_input = st.text_area(
        "AI의 역할과 동작을 설정해주세요:",
        value=st.session_state.system_prompt,
        height=150,
        key="system_prompt_input"
    )
    
    if st.button("프롬프트 적용", type="primary"):
        st.session_state.system_prompt = system_prompt_input
        st.success("시스템 프롬프트가 적용되었습니다!")
    
    st.divider()
    
    # 분석 대상 설정
    st.subheader("🔎 분석 대상 설정")
    
    # URL 입력
    url_input = st.text_input(
        "웹사이트 URL",
        placeholder="https://example.com",
        help="분석할 웹사이트의 URL을 입력하세요"
    )
    
    # 파일 업로드
    uploaded_files = st.file_uploader(
        "파일 업로드",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT 파일을 업로드할 수 있습니다"
    )
    
    # 파일 정보 표시
    if uploaded_files:
        total_size = sum(file.size for file in uploaded_files)
        size_mb = total_size / (1024 * 1024)
        st.info(f"📁 {len(uploaded_files)}개 파일, 총 {size_mb:.1f}MB")
        
        if size_mb > 10:
            st.warning("⚠️ 큰 파일입니다. 빠른 처리를 위해 기본 파서를 사용합니다.")
    
    # 분석 시작 버튼
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            if uploaded_files:
                status_text.text("📄 파일을 분석하고 있습니다...")
                progress_bar.progress(25)
                
                st.session_state.retriever = process_source("Files", uploaded_files)
                progress_bar.progress(100)
                
            elif url_input:
                status_text.text("🌐 URL을 분석하고 있습니다...")
                progress_bar.progress(25)
                
                st.session_state.retriever = process_source("URL", url_input)
                progress_bar.progress(100)
                
            else:
                st.warning("⚠️ 분석할 URL을 입력하거나 파일을 업로드해주세요.")
                progress_bar.empty()
                status_text.empty()
                
            if st.session_state.retriever:
                status_text.text("✅ 분석이 완료되었습니다!")
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                st.success("✅ 분석이 완료되었습니다! 이제 질문해보세요.")
            else:
                progress_bar.empty()
                status_text.empty()
                
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"분석 중 오류 발생: {e}")
    
    st.divider()
    
    # 성능 팁
    st.subheader("⚡ 성능 팁")
    st.info("""
    **빠른 처리를 위한 팁:**
    - 파일 크기는 10MB 이하 권장
    - 텍스트 파일(.txt)이 가장 빠름
    - 여러 파일보다 하나의 통합 파일 권장
    - 큰 PDF는 처리 시간이 오래 걸림
    """)
    
    st.divider()
    
    # 사용 팁
    st.subheader("💡 사용 팁")
    st.info("""
    **효과적인 질문 방법:**
    - 구체적이고 명확한 질문을 하세요
    - "어디에 나와 있나요?" 같은 출처 확인 질문도 유용합니다
    - 여러 관점에서 질문해보세요
    
    **예시 질문:**
    - "주요 내용을 요약해주세요"
    - "핵심 포인트는 무엇인가요?"
    - "이 문서의 결론은 무엇인가요?"
    """)
    
    # 사이드바 맨 아래에 대화 초기화 버튼
    st.markdown("---")
    if st.button("🔄 대화 초기화", type="secondary", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key not in ['system_prompt']:
                del st.session_state[key]
        
        st.session_state["messages"] = []
        st.session_state.retriever = None
        
        st.success("🔄 대화가 초기화되었습니다!")
        st.rerun()

# 메인 채팅 인터페이스
st.subheader("💬 채팅")

# 이전 메시지 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])

# 사용자 입력
user_input = st.chat_input("문서에 대해 궁금한 내용을 물어보세요! 🤔")

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
        st.chat_message("assistant").error(f"❌ 답변 생성 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()
