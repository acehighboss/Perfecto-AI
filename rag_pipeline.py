# rag_pipeline.py

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document as LangChainDocument
# --- 질의 확장을 위한 Import ---
from langchain_core.runnables import RunnablePassthrough
# ------------------------------------
from file_handler import get_documents_from_files

# get_retriever_from_source 함수는 이전과 동일하게 유지합니다.
def get_retriever_from_source(source_type, source_input):
    """
    URL 또는 파일로부터 문서를 로드하고, 텍스트를 분할하여
    Cohere Rerank가 적용된 ContextualCompressionRetriever를 생성합니다.
    """
    documents = []
    if source_type == "URL":
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            loader = WebBaseLoader(source_input, header_template=headers)
            documents = loader.load()
        except Exception as e:
            print(f"Error loading URL: {e}")
            return None
    elif source_type == "Files":
        llama_documents = get_documents_from_files(source_input)
        if not llama_documents:
            return None
        documents = [LangChainDocument(page_content=doc.text, metadata=doc.metadata) for doc in llama_documents]

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    if not splits:
        return None

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 10

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weights=[0.4, 0.6]
    )
    
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    
    return compression_retriever

# --- 이 함수를 수정합니다 ---
def get_conversational_rag_chain(retriever, system_prompt):
    """
    '질의 확장'을 포함하여 RAG 기능을 수행하는 체인을 생성합니다.
    """
    # 1. 질의 확장을 위한 프롬프트 정의
    query_expansion_template = """You are an AI assistant. Your task is to rephrase a user's question into a more detailed and specific search query for a vector database, based on the context of an AI industry report.
The expanded query should include related concepts, synonyms, and potential specific examples to improve document retrieval.

Original question:
{question}

Based on this question, generate a single, more descriptive search query in Korean.
Expanded Search Query:"""
    query_expansion_prompt = ChatPromptTemplate.from_template(query_expansion_template)
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 2. 질의 확장 체인 생성
    query_rewriter_chain = query_expansion_prompt | llm | StrOutputParser()

    # 3. 확장된 질의를 사용하여 문서를 검색하는 체인 생성
    #    사용자의 원본 질문(input)을 rewriter에 전달하여 확장된 검색어를 얻고, 그것으로 retriever를 호출합니다.
    context_retrieval_chain = (lambda x: x["input"]) | query_rewriter_chain | retriever
    
    # 4. 답변 생성을 위한 프롬프트 정의
    template = f"""{system_prompt}

Answer the user's request based on the provided "Context".
The "Context" is a collection of text snippets from a document or a URL.

**Instructions:**
1. First, carefully read the user's request and the provided "Context".
2. Find the answer within the "Context" and provide a clear, concise response based ONLY on this information.
3. Prioritize and cite only the most relevant sources that directly support your answer.
4. If the "Context" does not contain relevant information to fulfill the user's request, you must respond with: "죄송합니다. 제공된 내용만으로는 요청하신 작업을 수행할 수 없습니다."

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(template)
    
    # 5. 최종 RAG 체인 조립
    # RunnablePassthrough.assign을 사용하여 context를 동적으로 할당합니다.
    final_rag_chain = (
        RunnablePassthrough.assign(context=context_retrieval_chain)
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # create_retrieval_chain은 내부적으로 위와 유사한 로직을 수행합니다.
    # 질의 확장을 추가하기 위해 수동으로 체인을 조립하고, 답변 형식의 일관성을 위해 create_retrieval_chain과 유사하게 반환값을 구성합니다.
    # 다만, create_stuff_documents_chain의 반환 형식이 딕셔너리이므로 최종 체인의 출력은 StrOutputParser()로 문자열 답변만 반환하도록 합니다.
    # main.py에서 response['answer']가 아닌 response 자체를 사용하도록 조정이 필요합니다.
    
    # 더 나은 반환 구조를 위해 create_retrieval_chain을 재구성
    response_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    final_chain = RunnablePassthrough.assign(
        context=context_retrieval_chain,
    ).assign(
        answer=response_chain
    )
    
    return final_chain


# get_default_chain 함수는 이전과 동일하게 유지합니다.
def get_default_chain(system_prompt):
    """
    기본적인 대화형 체인을 생성합니다.
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
