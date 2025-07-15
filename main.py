import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline
import time

# 페이지 설정
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 지능형 문단/문장 분할 RAG 챗봇")
st.markdown(
    """
    **정확한 출처 기반 답변을 제공하는 RAG 챗봇입니다.**
    **문단 우선 분할**, **토큰 제한 고려**, **유사도 기반 출처 표시**로 더 정확한 답변을 제공합니다.
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
문서에 없는 정보라도 일반적인 지식으로 보충 설명할 수 있습니다."""

# 핸들러 및 파이프라인 초기화
@st.cache_resource
def initialize_components():
    file_handler = FileHandler()
    rag_pipeline = RAGPipeline()
    return file_handler, rag_pipeline

file_handler, rag_pipeline = initialize_components()

def process_source_with_progress(source_type, source_input):
    """진행 상황을 표시하며 소스 처리 - 미리보기 제거"""
    documents = []
    progress_placeholder = st.empty()
    
    try:
        if source_type == "URL":
            progress_placeholder.info("🌐 URL에서 콘텐츠를 가져오는 중...")
            documents = file_handler.get_documents_from_url(source_input)
            
        elif source_type == "Files":
            progress_placeholder.info("📄 파일을 읽는 중...")
            
            # 파일 크기 확인
            total_size = sum(len(file.getvalue()) for file in source_input)
            file_count = len(source_input)
            
            progress_placeholder.info(f"📊 {file_count}개 파일 (총 {total_size:,} bytes) 처리 중...")
            
            # LlamaParse 처리
            progress_placeholder.warning("⏳ LlamaParse로 문서를 분석하는 중...")
            documents = file_handler.get_documents_from_files(source_input)
        
        if not documents:
            progress_placeholder.error("❌ 문서를 추출하지 못했습니다.")
            return None
        
        # 문서 정보만 간단히 표시 (미리보기 제거)
        total_length = sum(len(doc.page_content) for doc in documents)
        st.info(f"📄 추출된 문서: {len(documents)}개, 총 {total_length:,}자")
        
        progress_placeholder.info("🔍 지능형 문단/문장 분할 검색기를 생성하는 중...")
        retriever = rag_pipeline.create_retriever(documents)
        
        if retriever:
            progress_placeholder.success("✅ 문서 분석 및 지능형 검색기 생성 완료!")
            time.sleep(1)
            progress_placeholder.empty()
            return retriever
        else:
            progress_placeholder.error("❌ 검색기 생성에 실패했습니다.")
            return None
            
    except Exception as e:
        progress_placeholder.error(f"❌ 처리 중 오류 발생: {str(e)}")
        return None

def display_smart_sources(source_documents):
    """유사도 기반 출처 표시 - 청크 타입과 토큰 수 정보 포함"""
    if source_documents:
        with st.expander("📚 유사도 기반 참고 출처"):
            for i, source in enumerate(source_documents):
                # 메타데이터 정보
                chunk_type = source.metadata.get('chunk_type', 'unknown')
                chunk_index = source.metadata.get('chunk_index', 'unknown')
                token_count = source.metadata.get('token_count', 0)
                source_file = source.metadata.get('source', 'unknown')
                
                st.text(f"--- 출처 {i+1} ({chunk_type}, 토큰: {token_count}) ---")
                st.caption(f"📁 {source_file} - 청크 인덱스: {chunk_index}")
                
                # 내용 표시 (너무 길면 축약)
                content = source.page_content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                st.markdown(content)
                st.divider()

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 지능형 분할 정보
    st.subheader("🧠 지능형 문서 분할")
    st.success("""
    **문단 우선 분할:**
    - 문단별 분할 우선 적용
    - 토큰 제한 초과 시 문장별 분할
    - 오버랩으로 맥락 보존
    - 유사도 기반 출처 표시
    """)
    
    st.divider()
    
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
        st.info(f"📁 업로드된 파일: {len(uploaded_files)}개")
        for file in uploaded_files:
            file_size = len(file.getvalue())
            st.write(f"• {file.name} ({file_size:,} bytes)")
    
    # 분석 시작 버튼
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retriever = None
        
        if uploaded_files:
            st.session_state.retriever = process_source_with_progress("Files", uploaded_files)
        elif url_input:
            st.session_state.retriever = process_source_with_progress("URL", url_input)
        else:
            st.warning("⚠️ 분석할 URL을 입력하거나 파일을 업로드해주세요.")
    
    st.divider()
    
    # 현재 상태 표시
    st.subheader("📊 현재 상태")
    if st.session_state.retriever:
        st.success("✅ 지능형 검색기 준비 완료!")
    else:
        st.info("⏳ 문서를 업로드하고 분석을 시작하세요")
    
    st.divider()
    
    # 사용 팁
    st.subheader("💡 지능형 분할 장점")
    st.info("""
    **개선된 분할 방식:**
    - 문단 구조 보존으로 맥락 유지
    - 토큰 제한 고려한 자동 분할
    - 문장별 오버랩으로 연결성 보장
    - 청크 타입별 최적화된 검색
    
    **출처 표시 개선:**
    - 유사도 기반 정확한 출처
    - 청크 타입과 토큰 수 정보
    - 문서 위치 정보 제공
    """)
    
    # 대화 초기화 버튼
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

# 분석 상태에 따른 안내 메시지
if not st.session_state.retriever:
    st.info("📋 먼저 왼쪽 사이드바에서 문서를 업로드하고 '분석 시작' 버튼을 클릭하세요.")

# 이전 메시지 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_smart_sources(message["sources"])

# 사용자 입력
user_input = st.chat_input("문서에 대해 궁금한 내용을 물어보세요! 🤔", disabled=not st.session_state.retriever)

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
                    st.info(f"🔍 지능형 분할 검색으로 {len(source_documents)}개의 유사도 높은 청크를 찾았습니다.")
                else:
                    st.warning("⚠️ 관련 문서를 찾지 못했지만 일반 지식으로 답변했습니다.")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": source_documents
                })
                
                display_smart_sources(source_documents)
        else:
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
