# src/evaluation_dataset_builder.py
import pandas as pd
import os


def build_eval_dataset(csv_path, sample_size=None):
    """
    sample_size: None이면 전체, 숫자면 그 개수만큼만 생성
    """
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding="cp949")

    # 샘플링 (질문 3개씩 생성되므로 행 개수 조절)
    if sample_size is not None:
        df = df.head(sample_size // 3 + 1)

    dataset = []

    for _, row in df.iterrows():
        title = row.get("사업명", "").strip()
        file_name = str(row.get('파일명', ''))  # 공통 변수로 뺌

        if not title:
            continue

        # [중요] 3번 질문을 위한 확장자 정답 생성 로직 (로더와 동일하게)
        _, ext_temp = os.path.splitext(file_name)
        clean_ext = ext_temp.lower().replace('.', '') if ext_temp else '알수없음'

        # 1. 예산 질문
        dataset.append({
            "source": file_name,  # 👈 [필수] 여기도 넣어야 예산 틀렸을 때 파일 확인 가능
            "question": f"{title}의 예산은 얼마인가?",
            "ground_truth": str(row.get("사업 금액", ""))
        })

        # 2. 발주 기관
        dataset.append({
            "source": file_name,  # 👈 [필수] 여기도 추가
            "question": f"{title}의 발주 기관은 어디인가?",
            "ground_truth": str(row.get("발주 기관", ""))
        })

        # 3. 문서 유형 (질문도 조금 더 명확하게 수정)
        dataset.append({
            "source": file_name,  # 👈 기존에 잘 넣으신 부분
            "question": f"'{file_name}' 문서의 파일 확장자는 무엇인가?",
            "ground_truth": clean_ext  # 👈 [수정] CSV 컬럼 대신 파일명에서 추출한 진짜 정답 사용
        })

    # 최종적으로 요청한 사이즈만큼만 반환
    return dataset[:sample_size] if sample_size else dataset