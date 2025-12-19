from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from typing import List


class RFPGenerator:
    def __init__(self):
        # 답변을 생성할 LLM 모델 설정
        self.llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

        # 프롬프트 템플릿 설정 (대화 기록을 위한 chat_history 추가)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 입찰 제안요청서(RFP) 분석 전문가 '입찰메이트'입니다. "
                       "주어진 [Context]를 바탕으로 사용자의 질문에 대해 명확하고 핵심적인 답변을 제공하세요. "
                       "이전 대화 맥락을 고려하여 자연스럽게 답변해야 합니다."
                       "만약 문서에 없는 내용이라면 '문서에서 관련 정보를 찾을 수 없습니다'라고 답하세요."),

            # 여기가 핵심! 이전 대화 내용이 이 자리에 들어갑니다.
            MessagesPlaceholder(variable_name="chat_history"),

            ("human", "질문: {question}\n\n[Context]:\n{context}")
        ])

        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_answer(self, query: str, retrieved_docs: List[Document], chat_history: List) -> str:
        """
        사용자 질문, 검색된 문서, 그리고 대화 기록을 바탕으로 답변을 생성합니다.
        """
        # 1. 검색된 문서들의 텍스트 합치기
        context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # 2. LLM에게 답변 요청 (chat_history 전달)
        answer = self.chain.invoke({
            "question": query,
            "context": context_text,
            "chat_history": chat_history
        })

        return answer