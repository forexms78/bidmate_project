import os
from dotenv import load_dotenv
from src.data_loader import RFPDataLoader

# 환경 변수 로드
load_dotenv()


def main():
    print("=== 입찰메이트 RAG 시스템 시작 ===")

    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드 중...")
    csv_path = os.path.join("DATA", "data_list.csv")

    loader = RFPDataLoader(file_path=csv_path)
    documents = loader.load()

    # 로드 결과 확인
    if documents:
        print(f"\n🎉 성공적으로 {len(documents)}개의 문서를 가져왔습니다.")
        sample_doc = documents[0]
        print(f"📌 샘플 문서: {sample_doc.metadata['title']}")
        print(f"📌 텍스트 길이: {len(sample_doc.page_content)} 자")
    else:
        print("🚨 문서를 가져오지 못했습니다. 경로와 CSV 파일을 확인해주세요.")


if __name__ == "__main__":
    main()