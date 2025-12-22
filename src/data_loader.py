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

            # 파일명에서 확장자 추출 (예: 입찰공고.hwp -> hwp)
            file_name = str(row.get('파일명', ''))
            file_ext = file_name.split('.')[-1] if '.' in file_name else '알수없음'

            # [핵심] 메타데이터를 본문에 강력하게 주입
            augmented_content = (
                f"[[문서 메타데이터]]\n"
                f"1. 사업명: {row.get('사업명', '무제')}\n"
                f"2. 발주 기관: {row.get('발주 기관', '알수없음')}\n"
                f"3. 사업 금액(예산): {row.get('사업 금액', '0')}원\n"
                f"4. 문서 형식(확장자): {file_ext}\n"  # <--- 이 부분이 정답률을 높입니다.
                f"5. 공고 번호: {row.get('공고 번호', '-')}\n"
                f"--------------------------------\n"
                f"[본문 내용]\n{content}"
            )

            metadata = {
                "source": file_name,
                "title": row.get('사업명', '무제'),
                "agency": row.get('발주 기관', '알수없음'),
                "extension": file_ext
            }

            doc = Document(page_content=augmented_content, metadata=metadata)
            docs.append(doc)

        print(f"✅ 데이터 로드 완료! (확장자 정보 포함됨)")
        return docs