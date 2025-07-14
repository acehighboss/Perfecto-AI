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
    
    # 문서 내용 확인 (디버깅용)
    if documents:
        total_length = sum(len(doc.page_content) for doc in documents)
        st.info(f"📄 추출된 문서: {len(documents)}개, 총 {total_length:,}자")
        
        # 첫 번째 문서의 일부 내용 표시 (확인용)
        if documents[0].page_content:
            preview = documents[0].page_content[:500] + "..." if len(documents[0].page_content) > 500 else documents[0].page_content
            with st.expander("📋 추출된 내용 미리보기"):
                st.text(preview)
        
        return rag_pipeline.create_retriever(documents)
    else:
        st.error("문서를 추출하지 못했습니다.")
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
        height=150,
        key="system_prompt_input"
    )
    
    # 시스템 프롬프트 적용 버튼 추가
    if st.button("🎯 페르소나 적용", type="primary", use_container_width=True):
        st.session_state.system_prompt = system_prompt_input
        st.success("✅ AI 페르소나가 적용되었습니다!")
        # 페르소나 변경 시 기존 대화는 유지하되, 다음 대화부터 새 페르소나 적용
    
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
    if st.button("🚀 분석 시작", type="primary", use_container_width=True):
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
        # 페르소나는 유지하고 대화만 초기화
        messages_backup = st.session_state.get("messages", [])
        system_prompt_backup = st.session_state.get("system_prompt", "")
        retriever_backup = st.session_state.get("retriever", None)
        
        st.session_state.clear()
        
        # 필요한 것만 복원
        st.session_state["messages"] = []
        st.session_state["system_prompt"] = system_prompt_backup
        st.session_state.retriever = None  # 대화 초기화 시 문서 분석도 초기화
        
        st.success("🔄 대화가 초기화되었습니다! (페르소나는 유지됨)")
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
                
                # 디버깅: 검색된 문서 수 표시
                if source_documents:
                    st.info(f"🔍 {len(source_documents)}개의 관련 문서를 찾았습니다.")
                else:
                    st.warning("⚠️ 관련 문서를 찾지 못했습니다.")
                
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
        st.chat_message("assistant").error(f"죄송합니다, 답변을 생성하는 중 오류가 발생했습니다.\n\n오류: {e}")
        st.session_state.messages.pop()
