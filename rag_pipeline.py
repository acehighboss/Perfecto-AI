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
from langchain_core.runnables import RunnableLambda

from file_handler import get_documents_from_files

# get_retriever_from_source 함수는 변경 없이 그대로 사용합니다.
def get_retriever_from_source(source_type, source_input):
    documents = []
    if source_type == "URL":
        try:
            loader = WebBaseLoader(source_input)
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


def get_conversational_rag_chain(retriever, system_prompt):
    """
    '역할 부여 및 사고 과정(Chain-of-Thought)' 프롬프트를 사용하여
    질의 재작성 기능이 포함된 RAG 체인을 생성합니다.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # 1. 일반화된 질의 재작성 프롬프트 (Zero-shot Chain-of-Thought)
    query_rewrite_template = """You are an expert at rewriting user questions into effective search queries for a vector database.
Your goal is to transform a user's question into a query that is rich with keywords and semantic details, based on the context of an AI industry trend report.
The rewritten query must be in Korean.

Follow these steps:
1.  **Analyze Intent**: Understand the core information the user is seeking.
2.  **Identify Key Concepts**: Extract main topics and entities (e.g., companies, technologies, people, countries).
3.  **Expand and Specify**: Brainstorm related, more specific terms.
    - If the user asks about a country's AI (e.g., "AI from China"), expand to major tech companies from that country (e.g., "Alibaba", "Tencent", "Baidu") and specific technologies (e.g., "LLM", "Tongyi Qianwen").
    - If the user asks about a concept (e.g., "AI safety"), expand to related terms (e.g., "AI alignment", "red-teaming", "AI ethics").
4.  **Construct a Descriptive Query**: Combine the original and expanded concepts into a single, comprehensive search query.

Original question: {question}
Rewritten Search Query:"""
    query_rewrite_prompt = ChatPromptTemplate.from_template(query_rewrite_template)
    
    # 2. 질의 재작성 체인
    query_rewriter = query_rewrite_prompt | llm | StrOutputParser()

    # 3. 질의 재작성 로직을 포함하는 '스마트 리트리버' 생성
    def rewrite_and_retrieve(query: str):
        print(f"Original Query: {query}")
        rewritten_query = query_rewriter.invoke({"question": query})
        print(f"Rewritten Query: {rewritten_query}")  # 디버깅을 위해 재작성된 쿼리 출력
        return retriever.invoke(rewritten_query)

    smart_retriever = RunnableLambda(rewrite_and_retrieve)

    # 4. 답변 생성을 위한 프롬프트
    rag_prompt_template = f"""{system_prompt}

Answer the user's request based on the provided "Context".

**Context:**
{{context}}

**User's Request:**
{{input}}
"""
    rag_prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    
    # 5. 문서를 결합하여 답변을 생성하는 체인
    document_chain = create_stuff_documents_chain(llm, rag_prompt)
    
    # 6. 표준 create_retrieval_chain을 사용하여 최종 체인 생성
    retrieval_chain = create_retrieval_chain(smart_retriever, document_chain)
    
    return retrieval_chain


def get_default_chain(system_prompt):
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{question}")]
    )
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    return prompt | llm | StrOutputParser()
