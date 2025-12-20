import pandas as pd
import os

def build_eval_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8", errors="ignore")
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
