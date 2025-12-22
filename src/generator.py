# @title src/generator.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RFPGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 공공 입찰 제안요청서(RFP) 전문 분석가입니다. 
            주어진 [Context]를 바탕으로 질문에 명확하게 답변하세요.

            [지침]
            1. **정확성:** 반드시 [Context]에 있는 내용만으로 답변하세요. 없는 내용은 "문서에서 정보를 찾을 수 없습니다"라고 하세요.
            2. **사업명 일치:** 질문에서 특정 사업(예: 봉화군 사업)을 물어봤다면, [Context] 내에서 **해당 사업명과 일치하는 메타데이터**를 최우선으로 신뢰하세요. 다른 사업의 정보가 섞여 있다면 무시하세요.
            3. **문서 유형 질문:** 사용자가 '문서 유형'이나 '형식'을 물으면, 내용적 특성(제안요청서 등)과 함께 **파일 확장자(hwp, pdf 등)**도 반드시 언급하세요. (예: "제안요청서(hwp)입니다.")
            4. **금액:** 금액은 원화(원) 단위로 정확히 표기하고, 필요시 괄호 안에 한글 금액을 병기하세요.
            """),
            ("human", """
            [Context]
            {context}

            [질문]
            {question}

            답변:
            """)
        ])

    def format_docs(self, docs):
        # 검색된 문서들의 내용을 합쳐서 프롬프트에 넣음
        return "\n\n".join([d.page_content for d in docs])

    def generate_answer(self, query, retrieved_docs, chat_history):
        # 체인 생성
        rag_chain = (
                {"context": lambda x: self.format_docs(retrieved_docs), "question": lambda x: query}
                | self.prompt
                | self.llm
                | StrOutputParser()
        )

        return rag_chain.invoke(query)