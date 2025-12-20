import pandas as pd
import os

def build_eval_dataset(csv_path):

    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        # utf-8로 읽기 실패 시 cp949(euc-kr)로 재시도
        df = pd.read_csv(csv_path, encoding="cp949")

    dataset = []

    for _, row in df.iterrows():
        title = row.get("사업명", "").strip()
        if not title:
            continue

        # 예산 질문
        dataset.append({
            "question": f"{title}의 예산은 얼마인가?",
            "ground_truth": str(row.get("사업 금액", ""))
        })

        # 발주 기관
        dataset.append({
            "question": f"{title}의 발주 기관은 어디인가?",
            "ground_truth": str(row.get("발주 기관", ""))
        })

        # 문서 유형
        dataset.append({
            "question": f"이 사업의 문서 유형은 무엇인가?",
            "ground_truth": str(row.get("파일형식", ""))
        })

    return dataset
