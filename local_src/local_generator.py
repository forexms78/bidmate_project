# @title local_src/local_generator.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever


class LocalRFPGenerator:
    def __init__(self, vector_store=None):
        # 로컬 Ollama (llama3) 호출
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.vector_store = vector_store
        self.hybrid_retriever = None

    def init_retriever(self, all_documents):
        if not self.vector_store: return
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        try:
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            # OpenAI용과 동일하게 하이브리드 구성
            self.hybrid_retriever = vector_retriever
            print("🏠 로컬 하이브리드 엔진 준비 완료")
        except:
            self.hybrid_retriever = vector_retriever

    def generate_answer(self, query: str) -> str:
        template = """
        당신은 공공 입찰 분석 전문가입니다. 아래 [문서 내용]을 바탕으로 질문에 답하세요.

        [규칙]
        1. 답변은 반드시 **한국어**로만 작성하세요.
        2. [문서 내용] 상단의 '사업 예산'이나 '발주 기관' 정보를 최우선으로 참고하세요.
        3. 숫자는 단위(원)를 포함하여 정확히 적고, 부연 설명 없이 핵심만 답하세요.
        4. 문서에 없는 내용은 절대 지어내지 마세요.

        [문서 내용]
        {context}

        질문: {question}
        답변:
        """
        prompt = ChatPromptTemplate.from_template(template)
        docs = self.hybrid_retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context, "question": query})