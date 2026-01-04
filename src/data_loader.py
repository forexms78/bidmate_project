# src/data_loader.py
# @title 수정된 RFPDataLoader 전체 코드
import os
import pandas as pd
from typing import List
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RFPDataLoader:
    def __init__(self, file_path: str):
        self.csv_path = file_path
        self.base_dir = os.path.dirname(file_path)
        self.files_dir = os.path.join(self.base_dir, "files")

        # 요약을 위한 LLM (빠르고 저렴한 gpt-4o-mini 사용 가정)
        # 모델명은 실제 사용 가능한 모델명으로 확인해주세요 (예: gpt-4o-mini, gpt-3.5-turbo 등)
        self.summary_llm = ChatOpenAI(model="gpt-5", temperature=0)

    def summarize_content(self, text: str, meta: dict) -> str:
        """
        긴 텍스트를 RAG에 넣기 좋게 핵심만 요약합니다.
        """
        if not text or len(text) < 100:  # 너무 짧으면 요약 안 함
            return text

        template = """
        당신은 공공 입찰 문서 전처리 전문가입니다. 
        아래 문서를 RAG 검색에 최적화되도록 핵심만 요약하세요.

        [필수 포함 정보]
        1. 문서 파일 형식: 원본 파일의 확장자가 {ext}임을 반드시 명시 (예: "이 문서는 hwp 파일입니다.")
        2. 사업명: {title}
        3. 예산 정보: 금액 관련 내용이 있다면 숫자와 단위(원)를 정확히 명시
        4. 발주 기관: {agency}
        5. 핵심 요약: 사업의 목적과 주요 과업 내용을 3줄 내외로 요약

        [원본 텍스트 일부]
        {text}

        [요약 결과]
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.summary_llm | StrOutputParser()

        try:
            # 텍스트가 너무 길면 앞부분 4000자만 잘라서 요약 (토큰 비용 절약)
            return chain.invoke({
                "text": text[:4000],
                "title": meta['title'],
                "agency": meta['agency'],
                "ext": meta['file_ext'] # 메타데이터 키 변경 반영
            })
        except Exception as e:
            print(f"⚠️ 요약 중 에러 발생 (건너뜀): {e}")
            return text[:500] + "..."  # 실패 시 원본 앞부분만 반환

    def load(self, use_summary: bool = False) -> List[Document]:
        """
        :param use_summary: True면 LLM을 통해 내용을 요약 후 저장합니다.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        try:
            df = pd.read_csv(self.csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.csv_path, encoding='cp949')

        all_docs = []
        print(f"📊 총 {len(df)}개의 데이터 처리를 시작합니다... (요약 모드: {'ON' if use_summary else 'OFF'})")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="문서 로딩 중"):
            file_name = str(row.get('파일명', ''))
            file_path = os.path.join(self.files_dir, file_name)

            # [수정됨] 1. 확장자 추출 로직 개선 (os.path 사용으로 정확도 향상)
            _, ext_temp = os.path.splitext(file_name)
            # 점(.)을 제거하고 소문자로 변환 (예: .HWP -> hwp)
            clean_ext = ext_temp.lower().replace('.', '') if ext_temp else '알수없음'

            # 2. 텍스트 추출 (clean_ext 사용)
            content = ""
            # PDF 처리
            if clean_ext == 'pdf' and os.path.exists(file_path):
                try:
                    loader = PyPDFLoader(file_path)
                    pages = loader.load()
                    content = "\n".join([p.page_content for p in pages])
                except Exception:
                    content = row.get('텍스트', '')
            # DOCX 처리
            elif clean_ext in ['docx', 'doc'] and os.path.exists(file_path):
                try:
                    loader = Docx2txtLoader(file_path)
                    pages = loader.load()
                    content = "\n".join([p.page_content for p in pages])
                except Exception:
                    content = row.get('텍스트', '')
            # HWP 또는 기타 (CSV 텍스트 사용)
            else:
                content = row.get('텍스트', '')

            # 내용 없으면 스킵
            if pd.isna(content) or str(content).strip() == "":
                continue

            # [수정됨] 3. 메타데이터 생성 (file_ext 추가)
            metadata = {
                "source": file_name,
                "title": row.get('사업명', '무제'),
                "agency": row.get('발주 기관', '알수없음'),
                "budget": row.get('사업 금액', 0),
                "file_ext": clean_ext, # ★ 핵심: 필터링을 위한 정확한 확장자 키
                "extension": clean_ext # (기존 코드 호환성을 위해 유지)
            }

            # 4. 요약 모드 적용 여부
            final_content = ""
            if use_summary:
                summary = self.summarize_content(content, metadata)
                final_content = f"[[AI 요약 정보]]\n{summary}\n\n================\n[[원본 상세 내용]]\n{content}"
            else:
                final_content = (
                    f"[[문서 정보]]\n"
                    f"- 파일형식: {clean_ext}\n" # clean_ext 사용
                    f"- 사업명: {metadata['title']}\n"
                    f"- 기관: {metadata['agency']}\n"
                    f"- 예산: {metadata['budget']}\n"
                    f"================\n{content}"
                )

            doc = Document(page_content=final_content, metadata=metadata)
            all_docs.append(doc)

        print(f"✅ 데이터 로드 완료! 총 {len(all_docs)}개 문서.")
        return all_docs