import streamlit as st
from file_handler import FileHandler
from rag_pipeline import RAGPipeline

# 페이지 설정
st.set_page_config(
    page_title="RAG 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'file_handler' not in st.session_state:
    st.session_state.file_handler = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "당신은 도움이 되는 AI 어시스턴트입니다."

def initialize_components():
    """컴포넌트 초기화"""
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        llama_api_key = st.secrets["LLAMA_API_KEY"]
    except KeyError as e:
        st.error(f"Streamlit secrets에서 {e} 키를 찾을 수 없습니다. secrets.toml 파일을 확인해주세요.")
        st.stop()
    
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline(google_api_key)
    
    if st.session_state.file_handler is None:
        st.session_state.file_handler = FileHandler(llama_api_key)

def main():
    st.title("🤖 RAG 챗봇")
    st.markdown("문서를 업로드하고 질문해보세요!")
    
    # 컴포넌트 초기화
    initialize_components()
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 1. 시스템 프롬프트 설정
        st.subheader("🎭 페르소나 설정")
        new_system_prompt = st.text_area(
            "시스템 프롬프트를 입력하세요:",
            value=st.session_state.system_prompt,
            height=100,
            help="챗봇의 성격과 답변 스타일을 설정할 수 있습니다."
        )
        
        if st.button("프롬프트 적용", type="primary"):
            st.session_state.system_prompt = new_system_prompt
            st.success("프롬프트가 적용되었습니다!")
        
        st.divider()
        
        # 2. 파일 업로드 섹션
        st.subheader("📁 문서 업로드")
        
        # URL 입력
        url_input = st.text_input("URL 입력:", placeholder="https://example.com")
        
        # 파일 업로드
        uploaded_files = st.file_uploader(
            "파일 업로드:",
            type=['pdf', 'docx', 'doc', 'txt'],
            accept_multiple_files=True,
            help="PDF, Word, 텍스트 파일을 업로드할 수 있습니다."
        )
        
        # 3. 분석 시작 버튼
        if st.button("📊 분석 시작", type="primary"):
            if not url_input and not uploaded_files:
                st.warning("URL 또는 파일을 입력해주세요.")
            else:
                process_documents(url_input, uploaded_files)
        
        # 4. 분석 상태 표시
        st.subheader("📈 분석 상태")
        doc_count = st.session_state.rag_pipeline.get_document_count() if st.session_state.rag_pipeline else 0
        
        if doc_count > 0:
            st.success(f"✅ 분석 완료 ({doc_count}개 청크)")
            st.session_state.documents_processed = True
        else:
            st.info("⏳ 문서를 업로드하고 분석을 시작해주세요.")
            st.session_state.documents_processed = False
        
        st.divider()
        
        # 5. 초기화 버튼
        st.subheader("🔄 초기화")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("대화 초기화", help="채팅 기록을 삭제합니다"):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("전체 초기화", help="모든 데이터를 삭제합니다"):
                st.session_state.messages = []
                if st.session_state.rag_pipeline:
                    st.session_state.rag_pipeline.clear_database()
                st.session_state.documents_processed = False
                st.rerun()
    
    # 메인 채팅 영역
    chat_container = st.container()
    
    with chat_container:
        # 채팅 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 입력
        if prompt := st.chat_input("질문을 입력하세요..."):
            if not st.session_state.documents_processed:
                st.warning("먼저 문서를 업로드하고 분석을 완료해주세요.")
                return
            
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("답변을 생성하고 있습니다..."):
                    # 유사 문서 검색
                    similar_docs = st.session_state.rag_pipeline.search_similar_documents(prompt)
                    
                    if similar_docs:
                        # 답변 생성
                        response = st.session_state.rag_pipeline.generate_answer(
                            prompt, 
                            similar_docs, 
                            st.session_state.system_prompt
                        )
                        st.markdown(response)
                        
                        # 참고 문서 표시
                        with st.expander("📚 참고한 문서들"):
                            for i, doc in enumerate(similar_docs, 1):
                                st.markdown(f"**[출처 {i}]** {doc['source']}")
                                st.markdown(f"```\n{doc['content'][:300]}...\n```")
                                st.markdown(f"*유사도: {doc['similarity']:.3f}*")
                                st.divider()
                    else:
                        response = "죄송합니다. 업로드된 문서에서 관련 정보를 찾을 수 없습니다."
                        st.markdown(response)
                
                # 응답을 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": response})

def process_documents(url_input, uploaded_files):
    """문서 처리 함수"""
    with st.spinner("문서를 분석하고 있습니다..."):
        success_count = 0
        total_count = 0
        
        # URL 처리
        if url_input:
            total_count += 1
            text = st.session_state.file_handler.extract_text_from_url(url_input)
            if text:
                chunks = st.session_state.file_handler.chunk_text(text)
                if st.session_state.rag_pipeline.add_documents(chunks, f"URL: {url_input}"):
                    success_count += 1
        
        # 파일 처리
        if uploaded_files:
            for uploaded_file in uploaded_files:
                total_count += 1
                text = st.session_state.file_handler.process_file(uploaded_file)
                if text:
                    chunks = st.session_state.file_handler.chunk_text(text)
                    if st.session_state.rag_pipeline.add_documents(chunks, f"파일: {uploaded_file.name}"):
                        success_count += 1
        
        if success_count > 0:
            st.success(f"✅ {success_count}/{total_count}개 문서가 성공적으로 처리되었습니다!")
            st.session_state.documents_processed = True
        else:
            st.error("❌ 문서 처리에 실패했습니다.")

if __name__ == "__main__":
    main()
