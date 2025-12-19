# @title src/vector_db.py
import os
import shutil
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List


class RFPVectorDB:
    def __init__(self, db_path: str = "./chroma_db"):
        """
        :param db_path: 벡터 DB가 저장될 로컬 경로
        """
        self.db_path = db_path
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")  # 가성비 좋은 모델
        self.vector_store = None

    def create_vector_db(self, documents: List[Document], force_rebuild: bool = False):
        """
        문서 리스트를 받아 청킹 후 벡터 DB를 생성 및 저장합니다.
        :param force_rebuild: True일 경우 기존 DB를 삭제하고 새로 만듭니다.
        """
        # 기존 DB가 있고 강제 재생성이 아니면 로드만 시도
        if os.path.exists(self.db_path) and not force_rebuild:
            print(f"📂 기존 벡터 DB를 불러옵니다. (경로: {self.db_path})")
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )
            return

        # DB 재생성 로직
        if force_rebuild and os.path.exists(self.db_path):
            print("🗑️ 기존 DB를 삭제하고 새로 생성합니다...")
            shutil.rmtree(self.db_path)  # 폴더 삭제

        print("✂️ 문서를 청킹(Chunking) 중입니다...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # 한 조각의 최대 길이
            chunk_overlap=200  # 조각 간 중복되는 길이 (문맥 끊김 방지)
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"🧩 청킹 완료! 총 {len(split_docs)}개의 청크가 생성되었습니다.")

        print("💾 벡터 DB에 저장 중입니다... (시간이 조금 걸릴 수 있습니다)")
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embedding_model,
            persist_directory=self.db_path
        )
        print("✅ 벡터 DB 저장 완료!")

    def get_retriever(self):
        """
        검색기(Retriever) 객체를 반환합니다.
        """
        if self.vector_store is None:
            # DB가 로드되지 않았다면 로드 시도
            self.vector_store = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model
            )

        # 검색 옵션 설정 (k=3: 가장 유사한 문서 3개 반환)
        return self.vector_store.as_retriever(search_kwargs={"k": 3})