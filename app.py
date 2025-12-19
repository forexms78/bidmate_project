import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from src.data_loader import RFPDataLoader
from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator


# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="입찰메이트 RAG",
    page_icon="📄",
    layout="wide"
)

st.title("🏢 입찰메이트 RAG 시스템")
st.caption("입찰 제안요청서(RFP) 기반 질의응답 시스템")


# ===============================
# 세션 상태 초기화
# ===============================
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.chat_history = []
    st.session_state.retriever = None
    st.session_state.generator = None


# ===============================
# 시스템 초기화 (1회만 실행)
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

        # 2. 벡터 DB 로드
        db_manager = RFPVectorDB(db_path="./chroma_db")
        db_manager.create_vector_db(documents, force_rebuild=False)
        retriever = db_manager.get_retriever()

        # 3. Generator
        generator = RFPGenerator()

        # 세션 저장
        st.session_state.retriever = retriever
        st.session_state.generator = generator
        st.session_state.initialized = True

    return True


# ===============================
# 초기화 버튼
# ===============================
if not st.session_state.initialized:
    st.info("📌 먼저 시스템을 초기화해주세요.")
    if st.button("🚀 시스템 초기화"):
        initialize_system()
    st.stop()


st.success("✅ 시스템 준비 완료!")
st.divider()


# ===============================
# 대화 UI
# ===============================
question = st.text_input(
    "🙋‍♂️ 질문을 입력하세요",
    placeholder="예: 그 사업 예산은 얼마야?"
)

if st.button("질문하기"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("🔍 답변 생성 중..."):
            retriever = st.session_state.retriever
            generator = st.session_state.generator

            # 1. 검색
            docs = retriever.invoke(question)

            # 2. 답변 생성
            answer = generator.generate_answer(
                query=question,
                retrieved_docs=docs,
                chat_history=st.session_state.chat_history
            )

        # 3. 대화 기록 저장
        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))

        # 길이 제한
        if len(st.session_state.chat_history) > 20:
            st.session_state.chat_history = st.session_state.chat_history[-20:]


# ===============================
# 대화 기록 출력
# ===============================
st.divider()
st.subheader("💬 대화 기록")

for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**🙋‍♂️ 질문:** {msg.content}")
    else:
        st.markdown(f"**🤖 답변:** {msg.content}")
