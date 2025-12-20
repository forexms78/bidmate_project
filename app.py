import os
import streamlit as st
import pandas as pd  # 데이터프레임 출력을 위해 추가
from langchain_core.messages import HumanMessage, AIMessage

from src.data_loader import RFPDataLoader
from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator
from src.evaluation import RFPEvaluator  # 평가 모듈 임포트

# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="입찰메이트 RAG",
    page_icon="📄",
    layout="wide"
)

st.title("🏢 입찰메이트 RAG 시스템")

# ===============================
# 세션 상태 초기화
# ===============================
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.generator = None


# ===============================
# 시스템 초기화 함수
# ===============================
def initialize_system():
    with st.spinner("📦 데이터 및 벡터 DB 로딩 중..."):
        csv_path = os.path.join("DATA", "data_list.csv")
        if not os.path.exists(csv_path):
            st.error("🚨 DATA/data_list.csv 파일을 찾을 수 없습니다.")
            return False

        # 1. 데이터 로드
        loader = RFPDataLoader(file_path=csv_path)
        documents = loader.load()

        # 2. 벡터 DB 로드 (force_rebuild=False로 하여 기존 DB 활용)
        db_manager = RFPVectorDB(db_path="./chroma_db")
        db_manager.create_vector_db(documents, force_rebuild=False)
        retriever = db_manager.get_retriever()

        # 3. Generator
        generator = RFPGenerator()

        st.session_state.retriever = retriever
        st.session_state.generator = generator
        st.session_state.initialized = True
    return True


# 초기화 버튼 (아직 초기화 안 된 경우)
if not st.session_state.initialized:
    st.info("📌 시스템을 시작하려면 아래 버튼을 눌러주세요.")
    if st.button("🚀 시스템 초기화"):
        initialize_system()
        st.rerun()  # 화면 새로고침
    st.stop()

# ===============================
# [사이드바] 관리자 메뉴 (성능 평가)
# ===============================
with st.sidebar:
    st.header("🛠️ 관리자 메뉴")
    st.write("RAG 모델의 성능을 평가합니다.")

    if st.button("📊 성능 평가 실행"):
        # 평가 객체 생성 (현재 로드된 retriever 재사용)
        evaluator = RFPEvaluator(retriever=st.session_state.retriever)

        # 프로그레스 바
        progress_bar = st.progress(0)
        status_text = st.empty()


        def update_progress(p, text):
            progress_bar.progress(p)
            status_text.text(text)


        # 평가 실행
        accuracy, results = evaluator.evaluate(progress_callback=update_progress)

        # 결과 표시
        st.success(f"평가 완료! 정확도: **{accuracy:.2f}%**")

        # 결과 데이터프레임으로 보여주기
        df_res = pd.DataFrame(results)
        st.dataframe(df_res, use_container_width=True)

        # 오답이 있다면 강조
        if accuracy < 100:
            st.warning("오답 노트를 확인하세요.")

# ===============================
# 메인화면: 대화 UI
# ===============================
st.caption("입찰 제안요청서(RFP) 기반 질의응답 챗봇입니다.")
st.divider()

# 대화 기록 출력
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    else:
        with st.chat_message("assistant"):
            st.write(msg.content)

# 입력창
if prompt := st.chat_input("질문을 입력하세요 (예: 부산국제영화제 사업 예산은?)"):
    # 사용자 메시지 표시
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            retriever = st.session_state.retriever
            generator = st.session_state.generator

            answer = generator.generate_answer(
                query=prompt,
                retrieved_docs=st.session_state.retriever.invoke(prompt),  # retriever 직접 호출
                chat_history=st.session_state.chat_history
            )
            st.write(answer)
    st.session_state.chat_history.append(AIMessage(content=answer))