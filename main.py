import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage  # 대화 기록용 메시지 객체

from src.data_loader import RFPDataLoader
from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator

# 환경 변수 로드
load_dotenv()


def main():
    print("\n" + "=" * 40)
    print("      🏢 입찰메이트 RAG 시스템 v1.1 (Memory)      ")
    print("=" * 40)

    # [1단계] 데이터 로드
    csv_path = os.path.join("DATA", "data_list.csv")
    if os.path.exists(csv_path):
        # 실제 로드가 필요하다면 아래 주석 해제 (여기선 생략 가능)
        loader = RFPDataLoader(file_path=csv_path)
        documents = loader.load()
    else:
        print("🚨 데이터 파일이 없습니다.")
        return

    # [2단계] 벡터 DB 로드
    print("📂 벡터 DB를 연결 중입니다...")
    db_manager = RFPVectorDB(db_path="./chroma_db")
    # 이미 구축된 DB 사용 (빠름)
    db_manager.create_vector_db(documents, force_rebuild=False)
    retriever = db_manager.get_retriever()

    # [3단계] 생성기 초기화
    generator = RFPGenerator()

    # [핵심] 대화 기록을 저장할 리스트 초기화
    chat_history = []

    print("\n🚀 시스템 준비 완료! (종료: 'exit')")
    print("💡 팁: '그 사업 예산은 얼마야?' 처럼 이어서 질문해보세요!")
    print("-" * 60)

    while True:
        user_input = input("\n🙋‍♂️ 질문: ").strip()

        if user_input.lower() in ['exit', 'quit', 'q', '종료']:
            print("👋 프로그램을 종료합니다.")
            break

        if not user_input:
            continue

        print("   🔍 답변 생성 중...")

        # 1. 검색 (Retrieval)
        relevant_docs = retriever.invoke(user_input)

        # 2. 답변 생성 (Generation) - chat_history 함께 전달!
        answer = generator.generate_answer(user_input, relevant_docs, chat_history)

        # 3. 결과 출력
        print("\n" + "=" * 60)
        print(f"🤖 입찰메이트 AI:\n{answer}")
        print("=" * 60)

        # 4. [핵심] 대화 기록 업데이트 (Human: 질문 / AI: 답변)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=answer))

        # (선택사항) 토큰 절약을 위해 대화 기록이 너무 길면 앞부분 삭제 (최근 10턴만 유지)
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]


if __name__ == "__main__":
    main()