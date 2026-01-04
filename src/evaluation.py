# @title src/evaluation.py
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class RFPEvaluator:
    def __init__(self, generator):
        self.generator = generator
        # 채점은 가장 똑똑하고 저렴한 모델로 수행
        self.judge_llm = ChatOpenAI(model="gpt-5", temperature=0)

    def _judge_answer(self, question, ground_truth, ai_answer):
        judge_template = """
        당신은 입찰 문서 분석 시스템의 유연한 채점관입니다.
        [질문]에 대한 [AI 답변]이 [실제 정답]과 맥락상 일치하거나, 질문의 의도를 더 잘 파악했다면 "정답" 처리하세요.

        [채점 가이드 - 필독]
        1. **숫자/금액**: 1억 3천만 원 = 130,000,000 = 1.3억 (모두 정답). 부가세 포함/별도 언급은 허용.
        2. **기관명**: '한영대학' = '한영대학교' (동일 기관이면 정답).
        3. **🚨 문서 유형 vs 확장자 (가장 중요)**:
           - 질문이 "문서 유형"을 묻는데 정답이 'hwp', 'pdf' 등 **확장자**인 경우:
             AI가 '제안요청서', 'RFP', '공고문' 등 **문서의 성격**을 맞게 대답했다면 **무조건 "정답"**으로 판정하세요.
           - 반대로 AI가 확장자(hwp)를 맞춰도 정답입니다. (둘 다 허용)
        4. **정보 부재**: 정답이 있는데 AI가 "모르겠다"고 하면 "오답".

        [데이터]
        - 질문: {question}
        - 실제 정답(Ground Truth): {ground_truth}
        - AI 답변: {ai_answer}

        판정 결과는 오직 "정답" 또는 "오답"으로만 출력하세요.
        판정:
        """
        prompt = ChatPromptTemplate.from_template(judge_template)
        chain = prompt | self.judge_llm | StrOutputParser()

        return chain.invoke({
            "question": question,
            "ground_truth": str(ground_truth),
            "ai_answer": ai_answer
        })

    def evaluate(self, dataset: pd.DataFrame, progress_callback=None):
        results = []
        correct_count = 0
        total = len(dataset)

        for i, row in dataset.iterrows():
            # 컬럼명 호환성 처리 (한글/영어)
            question = row.get('질문') or row.get('question')
            ground_truth = row.get('정답') or row.get('ground_truth')

            if not question: continue

            # AI 답변 생성
            ai_answer = self.generator.generate_answer(question)

            # 유연한 채점
            result_text = self._judge_answer(question, ground_truth, ai_answer)
            is_correct = "정답" in result_text

            if is_correct:
                correct_count += 1

            results.append({
                "질문": question,
                "정답": ground_truth,
                "AI 답변": ai_answer,
                "결과": "정답" if is_correct else "오답"
            })

            # 진행률 콜백
            if progress_callback:
                progress_callback((i + 1) / total, f"채점 중: {i + 1}/{total}")

        accuracy = (correct_count / total) * 100
        return accuracy, results