from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# 이 파일만 수정하면 됩니다.

def get_conversational_rag_chain(vector_store, system_prompt: str):
    """
    대화 내역과 질문 재작성을 모두 고려하는 가장 진보된 RAG 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 1. Retriever 생성 (main.py에서 k=5로 설정된 retriever를 그대로 사용)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # 2. "질문 재작성"을 위한 프롬프트와 체인 생성
    #    - 대화의 맥락을 보고, 후속 질문을 독립적인 검색 쿼리로 재작성합니다.
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. 재작성된 질문을 바탕으로 문서를 검색한 후, 원래 질문과 함께 답변을 생성하는 체인
    #    - 여기서 system_prompt (페르소나)가 사용됩니다.
    template = f"""{system_prompt}

    Answer the user's question based on the context provided below and the conversation history.
    The context may include text and tables in markdown format. You must be able to understand and answer based on them.
    If you don't know the answer, just say that you don't know. Don't make up an answer.

    Context:
    {{context}}
    """
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 4. 문서들을 하나의 컨텍스트로 결합하는 체인
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. history_aware_retriever와 Youtube_chain을 결합한 최종 RAG 체인
    return create_retrieval_chain(history_aware_retriever, Youtube_chain)


def get_default_chain(system_prompt: str):
    """문서 컨텍스트 없이 일반적인 대답을 하는 체인을 생성합니다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return prompt | llm | StrOutputParser()
