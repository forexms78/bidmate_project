# test_import.py
try:
    import rank_bm25

    print("✅ rank_bm25 설치됨")

    from langchain_community.retrievers import BM25Retriever

    print("✅ BM25Retriever 임포트 성공")

    from langchain.retrievers import EnsembleRetriever

    print("✅ EnsembleRetriever 임포트 성공")

except ImportError as e:
    print(f"❌ 임포트 실패: {e}")