# @title src/vector_db.py
import os
import shutil
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document


class RFPVectorDB:
    def __init__(self, db_path: str = "./chroma_db"):
        self.db_path = db_path
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = None

    def create_vector_db(self, documents: List[Document], force_rebuild: bool = False):
        # 1. 문서가 하나도 없으면 바로 중단 (에러 방지 핵심)
        if not documents:
            print("🚫 로드된 문서가 없습니다. DB 생성을 중단합니다.")
            return None

        # 2. 기존 DB 로드 시도
        if os.path.exists(self.db_path) and not force_rebuild:
            print(f"📂 기존 벡터 DB를 불러옵니다.")
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )
            return self.vector_store

        # 3. DB 폴더 삭제 및 재생성
        if os.path.exists(self.db_path):
            print("🗑️ 기존 DB 폴더 삭제 시도 중...")
            for _ in range(3):
                try:
                    shutil.rmtree(self.db_path)
                    print("✅ 기존 DB 삭제 성공")
                    break
                except PermissionError:
                    time.sleep(1)
            else:
                print("⚠️ 삭제 실패(파일 사용 중). 덮어쓰기를 시도합니다.")

        print("✂️ 문서를 청킹(Chunking) 중입니다...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)

        # [방어 코드] 청킹 결과가 비어있으면 중단
        if not split_docs:
            print("🚫 청킹된 문서가 없습니다 (내용이 비어있음).")
            return None

        print(f"💾 벡터 DB 생성 및 저장 중... (총 {len(split_docs)} 청크)")

        # 여기서 에러가 났던 부분입니다. 이제 split_docs가 있을 때만 실행됩니다.
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )
        return self.vector_store