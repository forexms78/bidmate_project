import pandas as pd
import os
from langchain_core.documents import Document
from typing import List

class RFPDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def load(self) -> List[Document]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")

        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.file_path, encoding='cp949')

        docs = []
        print(f"📊 데이터 로드 및 전처리 중... (총 {len(self.df)}행)")

        for idx, row in self.df.iterrows():
            content = row.get('텍스트', '')
            if pd.isna(content) or str(content).strip() == "":
                continue

            # [핵심 변경 사항] 메타데이터를 텍스트 본문에 '주입'합니다.
            # 이렇게 하면 "부산국제영화제 사업 찾아줘"라고 했을 때 검색이 훨씬 잘 됩니다.
            augmented_content = (
                f"문서 정보:\n"
                f"- 발주 기관: {row.get('발주 기관', '알수없음')}\n"
                f"- 사업명: {row.get('사업명', '무제')}\n"
                f"- 사업 금액: {row.get('사업 금액', '0')}원\n"
                f"- 공고 번호: {row.get('공고 번호', '-')}\n"
                f"\n[본문 내용]\n{content}"
            )

            metadata = {
                "source": row.get('파일명', 'unknown'),
                "title": row.get('사업명', '무제'),
                "agency": row.get('발주 기관', '알수없음'),
            }

            doc = Document(page_content=augmented_content, metadata=metadata)
            docs.append(doc)

        print(f"✅ 데이터 로드 완료! (메타데이터 주입됨)")
        return docs