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
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None

def initialize_components():
    """컴포넌트 초기화"""
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("Streamlit secrets에서 GOOGLE_API_KEY를 찾을 수 없습니다. secrets.toml 파일을 확인해주세요.")
        st.stop()
    
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline(google_api_key)
    
    if st.session_state.file_handler is None:
        st.session_state.file_handler = FileHandler()

def main():
    st.title("🤖 RAG 챗봇")
    st.markdown("문서를 업로드하고 질문해보세요! (LangChain 기반)")
    
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
            # RAG 체인 재생성
            if st.session_state.rag_pipeline and st.session_state.rag_pipeline.retriever:
                st.session_state.rag_chain = st.session_state.rag_pipeline.create_rag_chain(new_system_prompt)
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
        
        if st.session_state.rag_pipeline:
            vectorstore_info = st.session_state.rag_pipeline.get_vectorstore_info()
            doc_count = vectorstore_info.get("document_count", 0)
            
            if doc_count > 0:
                st.success(f"✅ 분석 완료 ({doc_count}개 청크)")
                st.info(f"임베딩 차원: {vectorstore_info.get('index_size', 0)}")
                st.session_state.documents_processed = True
            else:
                st.info("⏳ 문서를 업로드하고 분석을 시작해주세요.")
                st.session_state.documents_processed = False
        else:
            st.info("⏳ 시스템을 초기화하고 있습니다...")
        
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
                    st.session_state.rag_pipeline.clear_vectorstore()
                st.session_state.documents_processed = False
                st.session_state.rag_chain = None
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
                    # RAG 체인 사용 (더 효율적)
                    if st.session_state.rag_chain:
                        try:
                            response = st.session_state.rag_chain.invoke(prompt)
                        except Exception as e:
                            st.error(f"RAG 체인 실행 중 오류: {str(e)}")
                            response = "답변 생성 중 오류가 발생했습니다."
                    else:
                        # 기존 방식 사용
                        similar_docs = st.session_state.rag_pipeline.search_similar_documents(prompt)
                        if similar_docs:
                            response = st.session_state.rag_pipeline.generate_answer(
                                prompt, 
                                similar_docs, 
                                st.session_state.system_prompt
                            )
                        else:
                            response = "죄송합니다. 업로드된 문서에서 관련 정보를 찾을 수 없습니다."
                    
                    st.markdown(response)
                    
                    # 참고 문서 표시 (검색 기반)
                    similar_docs = st.session_state.rag_pipeline.search_similar_documents(prompt)
                    if similar_docs:
                        with st.expander("📚 참고한 문서들"):
                            for i, doc in enumerate(similar_docs, 1):
                                st.markdown(f"**[출처 {i}]** {doc['source']}")
                                st.markdown(f"```\n{doc['content'][:300]}...\n```")
                                
                                # 메타데이터 표시
                                metadata = doc.get('metadata', {})
                                if metadata:
                                    st.markdown(f"*메타데이터: {metadata}*")
                                st.divider()
                
                # 응답을 세션에 저장
                st.session_state.messages.append({"role": "assistant", "content": response})

def process_documents(url_input, uploaded_files):
    """문서 처리 함수"""
    all_documents = []
    success_count = 0
    total_count = 0
    
    with st.spinner("문서를 처리하고 있습니다..."):
        # URL 처리
        if url_input:
            total_count += 1
            st.info(f"URL 처리 중: {url_input}")
            documents = st.session_state.file_handler.load_url(url_input)
            if documents:
                all_documents.extend(documents)
                success_count += 1
                doc_info = st.session_state.file_handler.get_document_info(documents)
                st.success(f"URL 처리 완료: {doc_info['total_chunks']}개 청크 생성")
            else:
                st.error("URL에서 유효한 텍스트를 추출할 수 없습니다.")
        
        # 파일 처리
        if uploaded_files:
            for uploaded_file in uploaded_files:
                total_count += 1
                st.info(f"파일 처리 중: {uploaded_file.name}")
                documents = st.session_state.file_handler.load_file(uploaded_file)
                if documents:
                    all_documents.extend(documents)
                    success_count += 1
                    doc_info = st.session_state.file_handler.get_document_info(documents)
                    st.success(f"파일 처리 완료: {uploaded_file.name} ({doc_info['total_chunks']}개 청크)")
                else:
                    st.error(f"파일 처리 실패: {uploaded_file.name}")
    
    # 벡터스토어 생성
    if all_documents:
        st.info("벡터스토어를 생성하고 있습니다...")
        if st.session_state.rag_pipeline.create_vectorstore(all_documents):
            # RAG 체인 생성
            st.session_state.rag_chain = st.session_state.rag_pipeline.create_rag_chain(
                st.session_state.system_prompt
            )
            
            total_info = st.session_state.file_handler.get_document_info(all_documents)
            st.success(f"""
            ✅ 문서 처리 완료!
            - 처리된 문서: {success_count}/{total_count}개
            - 총 청크 수: {total_info['total_chunks']}개
            - 총 문자 수: {total_info['total_characters']:,}자
            - 출처: {', '.join(total_info['sources'])}
            """)
            st.session_state.documents_processed = True
        else:
            st.error("벡터스토어 생성에 실패했습니다.")
    else:
        st.error("처리할 수 있는 문서가 없습니다.")

if __name__ == "__main__":
    main()
