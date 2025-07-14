import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline

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
    
    # 분석 시작 버튼
if st.button("🚀 분석 시작", type="primary"):
    st.session_state.messages = []
    st.session_state.retriever = None
    
    if uploaded_files:
        with st.spinner("📄 파일을 분석하고 있습니다..."):
            st.session_state.retriever = process_source("Files", uploaded_files)
    elif url_input:
        with st.spinner("🌐 URL을 분석하고 있습니다..."):
            st.session_state.retriever = process_source("URL", url_input)
    else:
        st.warning("⚠️ 분석할 URL을 입력하거나 파일을 업로드해주세요.")

    if st.session_state.retriever:
        st.success("✅ 분석이 완료되었습니다! 이제 질문해보세요.")
