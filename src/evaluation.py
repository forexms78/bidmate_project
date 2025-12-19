import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tqdm import tqdm  # 진행률 표시바

# 기존 모듈 임포트
from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator

load_dotenv()

# ==========================================
# 1. 평가용 데이터셋 (Ground Truth) 준비
# 실제 데이터에 맞춰 질문과 정답을 늘려나가세요.
# ==========================================
TEST_DATASET = [
    {
        "question": "한영대학교 학사정보시스템 고도화 사업의 예산은 얼마인가?",
        "ground_truth": "130,000,000원 (1억 3천만 원)"
    },
    {
        "question": "부산국제영화제 온라인서비스 재개발 사업의 발주 기관은 어디인가?",
        "ground_truth": "(사)부산국제영화제"
    },
    {
        "question": "이 프로젝트에서 다루는 문서의 종류는 무엇인가?",
        "ground_truth": "제안요청서(RFP)"
    }
]


class RFPEvaluator:
    def __init__(self):
        # 채점관 모델 (Judge)
        self.judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

        # 시스템 모듈 로드
        self.db_manager = RFPVectorDB(db_path="./chroma_db")
        self.retriever = self.db_manager.get_retriever()
        self.generator = RFPGenerator()

    def evaluate(self):
        print(f"📊 총 {len(TEST_DATASET)}개의 문항에 대해 평가를 시작합니다...")

        score = 0
        results = []

        for item in tqdm(TEST_DATASET):
            question = item['question']
            truth = item['ground_truth']

            # 1. 우리 AI의 답변 생성
            relevant_docs = self.retriever.invoke(question)
            # 평가는 단발성 질문이므로 chat_history는 비워둡니다.
            prediction = self.generator.generate_answer(question, relevant_docs, chat_history=[])

            # 2. LLM 채점 (Judge)
            is_correct = self.judge_answer(question, truth, prediction)

            if is_correct:
                score += 1
                results.append("✅ 정답")
            else:
                results.append("❌ 오답")

            # 디버깅용 출력 (필요 시 주석 해제)
            # print(f"\nQ: {question}")
            # print(f"A(AI): {prediction}")
            # print(f"A(Truth): {truth}")
            # print(f"Result: {'Pass' if is_correct else 'Fail'}")

        # 최종 리포트
        accuracy = (score / len(TEST_DATASET)) * 100
        print("\n" + "=" * 30)
        print("      🏆 평가 결과 리포트      ")
        print("=" * 30)
        print(f"총 문항 수 : {len(TEST_DATASET)}")
        print(f"정답 수   : {score}")
        print(f"오답 수   : {len(TEST_DATASET) - score}")
        print(f"최종 정확도 : {accuracy:.2f}%")
        print("=" * 30)

    def judge_answer(self, question, truth, prediction):
        """
        AI 답변이 정답과 의미적으로 일치하는지 LLM에게 물어봅니다.
        """
        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 공정한 채점관입니다. [AI 답변]이 [정답]의 핵심 내용을 정확히 포함하고 있는지 판단하세요. "
                       "형식이 달라도 핵심 정보(숫자, 기관명 등)가 맞으면 정답입니다. "
                       "정답이면 'YES', 오답이면 'NO'라고만 대답하세요."),
            ("human", "질문: {question}\n정답: {truth}\nAI 답변: {prediction}")
        ])

        chain = judge_prompt | self.judge_llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "truth": truth,
            "prediction": prediction
        })

        return "YES" in result.upper()


if __name__ == "__main__":
    evaluator = RFPEvaluator()
    evaluator.evaluate()