import os
import sys

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tqdm import tqdm

from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator
from src.data_loader import RFPDataLoader

load_dotenv()

# 평가 데이터셋
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
    def __init__(self, retriever=None):
        """
        :param retriever: 이미 로드된 검색기가 있다면 재사용 (속도 향상)
        """
        self.judge_llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

        # 외부에서 retriever를 주입받으면 그것을 사용, 아니면 새로 구축
        if retriever:
            self.retriever = retriever
            self.db_manager = None  # 외부 주입시 불필요
        else:
            # 독립 실행 시 DB 로드 로직
            csv_path = os.path.join(root_dir, "DATA", "data_list.csv")
            loader = RFPDataLoader(file_path=csv_path)
            documents = loader.load()

            self.db_manager = RFPVectorDB(db_path=os.path.join(root_dir, "chroma_db"))
            self.db_manager.create_vector_db(documents, force_rebuild=True)
            self.retriever = self.db_manager.get_retriever()

        self.generator = RFPGenerator()

    def evaluate(self, progress_callback=None):
        """
        :param progress_callback: Streamlit 프로그레스 바 업데이트용 함수
        """
        score = 0
        results = []

        total = len(TEST_DATASET)

        for i, item in enumerate(TEST_DATASET):
            question = item['question']
            truth = item['ground_truth']

            # 진행률 업데이트 (Streamlit용)
            if progress_callback:
                progress_callback(i / total, f"평가 진행 중... ({i + 1}/{total})")

            # 1. 답변 생성
            relevant_docs = self.retriever.invoke(question)
            prediction = self.generator.generate_answer(question, relevant_docs, chat_history=[])

            # 2. 채점
            is_correct = self.judge_answer(question, truth, prediction)

            result_item = {
                "질문": question,
                "정답": truth,
                "AI 답변": prediction,
                "채점 결과": "✅ 정답" if is_correct else "❌ 오답"
            }
            results.append(result_item)

            if is_correct:
                score += 1

        # 완료 시 진행률 100%
        if progress_callback:
            progress_callback(1.0, "평가 완료!")

        accuracy = (score / total) * 100
        return accuracy, results

    def judge_answer(self, question, truth, prediction):
        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 채점관입니다. [AI 답변]이 [정답]의 핵심 정보(기관명, 숫자 등)를 포함하면 'YES', 아니면 'NO'를 출력하세요."),
            ("human", "질문: {question}\n정답: {truth}\nAI 답변: {prediction}")
        ])
        chain = judge_prompt | self.judge_llm | StrOutputParser()
        result = chain.invoke({"question": question, "truth": truth, "prediction": prediction})
        return "YES" in result.upper()


if __name__ == "__main__":
    # 터미널에서 단독 실행 시
    evaluator = RFPEvaluator()
    acc, res = evaluator.evaluate()
    print(f"정확도: {acc}%")