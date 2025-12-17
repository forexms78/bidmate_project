import pandas as pd
import os
from langchain_core.documents import Document
from typing import List, Optional


class RFPDataLoader:
    def __init__(self, file_path: str):
        """
        :param file_path: 메타데이터 및 텍스트가 포함된 CSV 파일 경로 (예: DATA/data_list.csv)
        """
        self.file_path = file_path
        self.df = None

    def load(self) -> List[Document]:
        """
        CSV 파일을 읽어 LangChain Document 리스트로 반환합니다.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")

        # CSV 읽기 (인코딩 에러 처리)
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            self.df = pd.read_csv(self.file_path, encoding='cp949')

        docs = []

        print(f"📊 총 {len(self.df)}개의 행을 처리합니다...")

        for idx, row in self.df.iterrows():
            # 1. 텍스트 내용 가져오기 (비어있으면 건너뜀)
            content = row.get('텍스트', '')
            if pd.isna(content) or str(content).strip() == "":
                print(f"⚠️ 경고: {idx}번 행의 텍스트가 비어있습니다. (파일명: {row.get('파일명')})")
                continue

            # 2. 메타데이터 구성 (RAG 검색 시 필터링에 사용할 정보들)
            # 금액 같은 숫자는 문자열로 처리하거나 전처리 필요할 수 있음
            metadata = {
                "source": row.get('파일명', 'unknown'),
                "title": row.get('사업명', '무제'),
                "agency": row.get('발주 기관', '알수없음'),
                "category": row.get('파일형식', 'hwp'),
                "budget": row.get('사업 금액', 0),
                "notice_no": row.get('공고 번호', ''),
                "date": row.get('공개 일자', '')
            }

            # 3. Document 객체 생성
            # page_content는 실제 임베딩할 텍스트, metadata는 부가 정보
            doc = Document(page_content=str(content), metadata=metadata)
            docs.append(doc)

        print(f"✅ 데이터 로드 완료! 총 {len(docs)}개의 문서 객체가 생성되었습니다.")
        return docs


# 테스트용 코드 (이 파일만 실행했을 때 동작)
if __name__ == "__main__":
    loader = RFPDataLoader(file_path="../DATA/data_list.csv")  # 경로 주의 (실행 위치에 따라 다름)
    documents = loader.load()

    if documents:
        print("\n=== 첫 번째 문서 미리보기 ===")
        print(f"🔹 파일명: {documents[0].metadata['source']}")
        print(f"🔹 내용(앞 200자): {documents[0].page_content[:200]}...")