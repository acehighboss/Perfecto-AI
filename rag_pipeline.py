from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# [추가] 압축 검색에 필요한 모듈 import
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

def get_conversational_rag_chain(vector_store, system_prompt: str):
    """
    고급 압축 검색(Contextual Compression Retriever)을 포함한 대화 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 1. 기본 Retriever 설정: 더 많은 후보군(k=10)을 가져오도록 설정
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. Compressor 설정: LLM을 사용하여 관련성 높은 문장만 추출
    compressor = LLMChainExtractor.from_llm(llm)

    # 3. 압축 검색기(Compression Retriever) 생성
    #    - 1차로 base_retriever가 문서를 가져오면,
    #    - 2차로 compressor가 관련성 높은 내용만 압축/필터링합니다.
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # 4. RAG 프롬프트 설정 (기존과 동일)
    template = f"""{system_prompt}

Answer the user's question based on the context provided below and the conversation history.
The context may include text and tables in markdown format. You must be able to understand and answer based on them.
If you don't know the answer, just say that you don't know. Don't make up an answer.

Context:
{{context}}
"""
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    # 5. 최종 체인 생성 (Retriever를 compression_retriever로 교체)
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    return create_retrieval_chain(compression_retriever, document_chain)


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
