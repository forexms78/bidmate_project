# @title src/generator.py (출력 형식 최적화)
import sys
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever


# 라이브러리 의존성 없이 작동하는 하이브리드 검색기
class SimpleHybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever

    def invoke(self, query):
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        except:
            bm25_docs = []
        vector_docs = self.vector_retriever.invoke(query)

        combined_docs = []
        seen_content = set()
        for doc in bm25_docs + vector_docs:
            if doc.page_content not in seen_content:
                combined_docs.append(doc)
                seen_content.add(doc.page_content)
        return combined_docs[:5]


class RFPGenerator:
    def __init__(self, vector_store=None):
        self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
        self.vector_store = vector_store
        self.hybrid_retriever = None

    def init_retriever(self, all_documents):
        if not self.vector_store: return
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        if not all_documents:
            self.hybrid_retriever = vector_retriever
            return
        try:
            bm25_retriever = BM25Retriever.from_documents(all_documents)
            self.hybrid_retriever = SimpleHybridRetriever(vector_retriever, bm25_retriever)
            print("🚀 하이브리드 검색기 가동")
        except:
            self.hybrid_retriever = vector_retriever

    def generate_answer(self, query: str, chat_history: list = None) -> str:
        if not self.hybrid_retriever: return "검색기 미초기화"

        # [수정 포인트] 지시 사항을 더 명확하게 변경
        template = """
        당신은 공공 입찰 제안요청서(RFP) 분석 전문가입니다. 
        반드시 아래 [Context]의 내용만을 근거로 답변하되, 다음 규칙을 엄격히 지키세요.

        [답변 프로세스 및 규칙]
        1. **메타데이터 우선 참조:** 답변을 구성하기 전, [Context] 상단의 '[[AI 요약 정보]]' 섹션에 있는 '사업명', '발주 기관', '사업 예산' 항목을 본문보다 먼저 확인하고 최우선 정보로 신뢰하세요.
        2. **금액 정보:** 본문에 상세 금액이 없더라도, 요약 정보에 금액이 명시되어 있다면 해당 금액을 정답으로 채택하여 답변하세요. 금액은 반드시 원화(원) 단위를 포함하여 정확히 표기합니다.
        3. **확장자 노출 금지 (중요):** 답변 끝이나 중간에 '(문서 형식: hwp)', '(확장자: pdf)'와 같은 메타 정보를 **절대 추가하지 마세요.** 4. **예외적 확장자 답변:** 사용자가 질문에서 직접적으로 "파일 형식이 무엇인가요?" 또는 "확장자가 무엇인가요?"라고 물었을 때만 해당 정보를 답변에 포함하세요. 그 외의 질문에는 언급하지 않습니다.
        5. **무관한 답변 지양:** 질문에 대한 핵심 정보만 간결하고 명확하게 답변하세요. [Context]에서 정보를 찾을 수 없는 경우에만 "정보를 찾을 수 없습니다"라고 답변하세요.

        [Context]
        {context}

        [질문]
        {question}

        답변:
        """
        prompt = ChatPromptTemplate.from_template(template)

        try:
            retrieved_docs = self.hybrid_retriever.invoke(query)
            context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
            chain = prompt | self.llm | StrOutputParser()
            return chain.invoke({"question": query, "context": context_text})
        except Exception as e:
            return f"오류 발생: {str(e)}"