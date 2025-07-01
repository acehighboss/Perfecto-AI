import streamlit as st
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# API KEY를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv

# API KEY 정보로드
load_dotenv()

st.set_page_config(page_title="RAG-enhanced Gemini Chatbot", page_icon="🔗")
st.title("🔗 RAG 기능이 추가된 Gemini 챗봇")
st.markdown(
    """
안녕하세요! 이 챗봇은 일반적인 대화뿐만 아니라, 웹사이트 내용을 분석하고 해당 내용을 기반으로 질문에 답변하는 RAG 기능도 갖추고 있습니다.

**사용 방법:**
1.  왼쪽 사이드바에서 분석하고 싶은 웹사이트의 전체 URL을 입력하세요.
2.  '사이트 분석 시작' 버튼을 클릭하세요. (분석에는 잠시 시간이 걸릴 수 있습니다.)
3.  분석이 완료되면, 해당 웹사이트 내용에 대해 질문해보세요!
"""
)


# --- RAG 파이프라인 관련 함수 ---
@st.cache_resource(show_spinner="웹사이트를 분석 중입니다...")
def get_retriever_from_url(url):
    """
    URL로부터 문서를 로드하고, 텍스트를 분할하여 벡터 저장소 기반의 retriever를 생성합니다.
    Streamlit의 cache_resource 데코레이터를 사용하여 한 번 생성된 retriever는 세션 내에서 재사용됩니다.
    """
    # 1. 문서 로드
    loader = WebBaseLoader(url)
    documents = loader.load()

    # 2. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)

    # 3. 임베딩 및 벡터 저장소 생성
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)

    # 4. Retriever 생성
    # # 가장 유사한 문서 3개를 가져오도록 k값 설정
    # return vectorstore.as_retriever(search_kwargs={"k": 3})
    # 유사도 점수 0.5 이상인 문서만 가져오도록 설정
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
    )


# --- LangChain 체인 생성 함수 ---
def get_conversational_rag_chain(retriever):
    """
    RAG 기능을 수행하는 체인을 생성합니다.
    """
    # RAG 프롬프트 정의
    rag_prompt = ChatPromptTemplate.from_template(
        """
        Answer the user's question based on the context provided below.
        If you don't know the answer, just say that you don't know. Don't make up an answer.

        Context:
        {context}

        Question:
        {input}
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 문서들을 하나의 문자열로 결합하는 체인
    document_chain = create_stuff_documents_chain(llm, rag_prompt)

    # 검색된 문서를 document_chain으로 전달하는 체인
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def get_default_chain():
    """
    기본적인 대화형 체인을 생성합니다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절한 AI 어시스턴트입니다."),
            ("user", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# --- 사이드바 UI ---
with st.sidebar:
    st.header("RAG 설정")
    url_input = st.text_input(
        "분석할 웹사이트 URL을 입력하세요", placeholder="https://example.com"
    )

    if st.button("사이트 분석 시작"):
        if url_input:
            st.session_state.retriever = get_retriever_from_url(url_input)
            st.success("사이트 분석이 완료되었습니다! 이제 질문해보세요.")
        else:
            st.warning("URL을 입력해주세요.")

    st.divider()
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.session_state.retriever = None
        st.rerun()

# --- 메인 채팅 화면 ---
# 이전 대화 기록 출력
for chat_message in st.session_state["messages"]:
    st.chat_message(chat_message.role).write(chat_message.content)

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    st.chat_message("user").write(user_input)

    # RAG 기능이 활성화되었는지 확인
    if st.session_state.retriever:
        chain = get_conversational_rag_chain(st.session_state.retriever)
        # RAG 체인의 입력 형식은 {"input": user_question} 입니다.
        response = chain.stream({"input": user_input})

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            # RAG 체인의 응답 구조가 다르므로, 'answer' 키를 확인합니다.
            for chunk in response:
                if "answer" in chunk:
                    ai_answer += chunk["answer"]
                    container.markdown(ai_answer)
    else:
        # RAG 기능이 비활성화된 경우, 기본 체인 사용
        chain = get_default_chain()
        response = chain.stream({"question": user_input})

        with st.chat_message("assistant"):
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

    # 대화기록 저장
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    st.session_state.messages.append(ChatMessage(role="assistant", content=ai_answer))
