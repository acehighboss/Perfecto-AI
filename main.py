import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline

# 페이지 설정
st.set_page_config(page_title="Universal Table RAG Chatbot", page_icon="📊", layout="wide")
st.title("📊RAG 챗봇")
st.markdown(
    """
    **LlamaParser를 활용한 RAG 챗봇입니다.**
    """
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """당신은 문서 분석 전문가 AI 어시스턴트입니다. 
주어진 문서의 텍스트, 테이블, 이미지 내용을 정확히 이해하고 상세하게 답변해주세요.
답변할 때는 반드시 참조한 출처를 명시하고, 정확한 정보만을 제공해주세요."""
if "document_type" not in st.session_state:
    st.session_state.document_type = "general"

# 핸들러 및 파이프라인 초기화
@st.cache_resource
def initialize_components():
    file_handler = FileHandler()
    rag_pipeline = RAGPipeline()
    return file_handler, rag_pipeline

file_handler, rag_pipeline = initialize_components()

def process_source(source_type, source_input, document_type):
    """소스 처리 및 검색기 생성"""
    documents = []
    
    if source_type == "URL":
        documents = file_handler.get_documents_from_url(source_input)
    elif source_type == "Files":
        documents = file_handler.get_documents_from_files(source_input, document_type)
    
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
    
    # 문서 타입 선택
    st.subheader("📋 문서 타입 선택")
    document_type_options = {
        "general": "🔍 일반 문서",
        "financial": "💰 재무/회계",
        "research": "🔬 연구/실험",
        "inventory": "📦 재고/물류",
        "hr": "👥 인사/조직",
        "sales": "📈 영업/마케팅"
    }
    
    selected_type = st.selectbox(
        "문서 타입을 선택하세요:",
        options=list(document_type_options.keys()),
        format_func=lambda x: document_type_options[x],
        index=0
    )
    st.session_state.document_type = selected_type
    
    # 선택된 타입에 대한 설명
    type_descriptions = {
        "general": "모든 종류의 일반적인 표와 데이터를 처리합니다.",
        "financial": "재무제표, 손익계산서, 예산 등 재무 관련 표를 최적화하여 처리합니다.",
        "research": "실험 결과, 통계 데이터, 연구 보고서의 표를 정확하게 해석합니다.",
        "inventory": "재고 현황, 입출고 내역, 물류 데이터를 체계적으로 분석합니다.",
        "hr": "직원 정보, 급여 데이터, 평가 결과 등 인사 관련 표를 처리합니다.",
        "sales": "매출 실적, 고객 데이터, 영업 성과 등을 분석합니다."
    }
    
    st.info(f"💡 {type_descriptions[selected_type]}")
    
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
        type=["pdf", "docx", "txt", "xlsx", "csv"],
        accept_multiple_files=True,
        help="PDF, DOCX, TXT, XLSX, CSV 파일을 업로드할 수 있습니다"
    )
    
    # 분석 시작 버튼
    if st.button("🚀 분석 시작", type="primary"):
        st.session_state.messages = []
        st.session_state.retriever = None
        
        if uploaded_files:
            with st.spinner(f"📄 {document_type_options[selected_type]} 문서를 분석하고 있습니다..."):
                st.session_state.retriever = process_source("Files", uploaded_files, selected_type)
        elif url_input:
            with st.spinner("🌐 URL을 분석하고 있습니다..."):
                st.session_state.retriever = process_source("URL", url_input, selected_type)
        else:
            st.warning("⚠️ 분석할 URL을 입력하거나 파일을 업로드해주세요.")

        if st.session_state.retriever:
            st.success("✅ 분석이 완료되었습니다! 이제 질문해보세요.")
    
    st.divider()
    
    # 대화 초기화
    if st.button("🔄 대화 초기화"):
        st.session_state.clear()
        st.rerun()

# 메인 채팅 인터페이스
st.subheader("💬 채팅")

# 이전 메시지 표시
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            display_sources(message["sources"])
        if "validation" in message and message["validation"]:
            st.warning(message["validation"]["warning"])
            st.info(message["validation"]["suggestion"])

# 사용자 입력
user_input = st.chat_input("표나 데이터에 대해 궁금한 내용을 물어보세요! 📊")

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
                
                # 답변 검증
                validation_result = rag_pipeline.validate_table_response(
                    user_input, ai_answer, source_documents
                )
                
                # 메시지 저장
                message_data = {
                    "role": "assistant", 
                    "content": ai_answer, 
                    "sources": source_documents
                }
                
                if validation_result:
                    message_data["validation"] = validation_result
                    st.warning(validation_result["warning"])
                    st.info(validation_result["suggestion"])
                
                st.session_state.messages.append(message_data)
                
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
        st.session_state.messages.pop()  # 오류 발생 시 마지막 메시지 제거

# 푸터
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        📊 Universal Table RAG Chatbot - 모든 종류의 표와 데이터를 정확하게 분석합니다
    </div>
    """, 
    unsafe_allow_html=True
)
