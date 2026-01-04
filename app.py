# @title app.py (최종 수정본)
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from langchain_core.messages import HumanMessage, AIMessage

# 모듈 임포트
from src.data_loader import RFPDataLoader
from src.vector_db import RFPVectorDB
from src.generator import RFPGenerator
from local_src.local_generator import LocalRFPGenerator
from src.evaluation import RFPEvaluator
from src.evaluation_dataset_builder import build_eval_dataset

# 그래프 한글 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="입찰메이트 통합 RAG", page_icon="🏢", layout="wide")

# ==========================================
# [세션 상태 초기화]
# ==========================================
if "initialized_b" not in st.session_state:
    st.session_state.update({
        "initialized_b": False, "history_b": [], "gen_b": None,
        "initialized_a": False, "history_a": [], "gen_a": None,
        "comparison_results": None, "acc_b": 0, "acc_a": 0, "docs": None,
        "load_error_b": None, "load_error_a": None
    })


# ==========================================
# [자동 엔진 로드]
# ==========================================
def auto_initialize():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "DATA", "data_list.csv")
    db_path_b = os.path.join(base_dir, "chroma_db")
    db_path_a = os.path.join(base_dir, "local_chroma_db")

    if not os.path.exists(csv_path): return

    # 1. GPT 엔진 자동 로드
    if not st.session_state.initialized_b and os.path.exists(db_path_b):
        try:
            with st.spinner("GPT 엔진 연결 중..."):
                loader = RFPDataLoader(file_path=csv_path)
                docs = loader.load(use_summary=False)
                db = RFPVectorDB(db_path=db_path_b)
                store = db.create_vector_db(docs, force_rebuild=False)
                gen_b = RFPGenerator(store)
                gen_b.init_retriever(docs)
                st.session_state.initialized_b = True
                st.session_state.gen_b = gen_b
                st.session_state.docs = docs
                st.session_state.load_error_b = None
        except Exception as e:
            st.session_state.load_error_b = str(e)

    # 2. 로컬 엔진 자동 로드
    if not st.session_state.initialized_a and os.path.exists(db_path_a):
        try:
            loader = RFPDataLoader(file_path=csv_path)
            docs = loader.load(use_summary=False)
            db = RFPVectorDB(db_path=db_path_a)
            store = db.create_vector_db(docs, force_rebuild=False)
            gen_a = LocalRFPGenerator(store)
            gen_a.init_retriever(docs)
            st.session_state.initialized_a = True
            st.session_state.gen_a = gen_a
            st.session_state.docs = docs
            st.session_state.load_error_a = None
        except Exception as e:
            st.session_state.load_error_a = str(e)


auto_initialize()

st.title("🏢 입찰메이트 통합 RAG 대시보드")

# ==========================================
# [사이드바 UI]
# ==========================================
with st.sidebar:
    st.header("⚙️ 엔진 상태")

    # GPT 상태
    if st.session_state.initialized_b:
        st.success("API: 🟢 가동됨")
    else:
        st.error("API: 🔴 중단됨")
        if st.session_state.load_error_b:
            st.caption(f"⚠️ {st.session_state.load_error_b}")
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")):
            if st.button("🔌 API 긴급 연결"):
                auto_initialize()
                st.rerun()

    # 로컬 상태
    if st.session_state.initialized_a:
        st.success("로컬: 🟢 가동됨")
    else:
        st.error("로컬: 🔴 중단됨")
        if st.session_state.load_error_a:
            st.caption(f"⚠️ {st.session_state.load_error_a}")
        if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_chroma_db")):
            if st.button("🔌 로컬 긴급 연결"):
                auto_initialize()
                st.rerun()

    st.divider()
    st.header("🛠️ Vector DB 구축")
    use_summary = st.checkbox("문서 요약을 통한 구축", value=True)

    col_b, col_a = st.columns(2)
    with col_b:
        if st.button("API 구축"):
            with st.spinner("GPT DB 재구축 중..."):
                loader = RFPDataLoader(file_path=os.path.join("DATA", "data_list.csv"))
                docs = loader.load(use_summary=use_summary)
                db = RFPVectorDB(db_path="./chroma_db")
                store = db.create_vector_db(docs, force_rebuild=True)
                gen_b = RFPGenerator(store)
                gen_b.init_retriever(docs)
                st.session_state.update({"gen_b": gen_b, "initialized_b": True})
            st.success("완료")
            st.rerun()

    with col_a:
        if st.button("로컬 구축"):
            with st.spinner("로컬 DB 재구축 중..."):
                loader = RFPDataLoader(file_path=os.path.join("DATA", "data_list.csv"))
                docs = loader.load(use_summary=False)
                db = RFPVectorDB(db_path="./local_chroma_db")
                store = db.create_vector_db(docs, force_rebuild=True)
                gen_a = LocalRFPGenerator(store)
                gen_a.init_retriever(docs)
                st.session_state.update({"gen_a": gen_a, "initialized_a": True})
            st.success("완료")
            st.rerun()

    st.divider()
    st.header("📊 A vs B 성능 비교 평가")
    eval_count = st.number_input("평가 문항 수 (MAX 100)", min_value=1, max_value=100, value=5)

    # --- [수정 완료] 버튼 및 평가 로직 ---
    if st.button("🏁 성능 비교 시작"):
        if st.session_state.initialized_a and st.session_state.initialized_b:
            progress_bar = st.progress(0)
            status_text = st.empty()


            def update_progress(percent, msg):
                progress_bar.progress(percent)
                status_text.write(msg)


            try:
                # 1. 데이터 생성
                status_text.write("📚 평가 데이터 생성 중...")
                raw_data = build_eval_dataset(os.path.join("DATA", "data_list.csv"), sample_size=eval_count)
                test_ds = pd.DataFrame(raw_data)

                # 2. GPT-5 평가
                status_text.write("🤖 GPT-5 채점 시작...")
                eval_b = RFPEvaluator(st.session_state.gen_b)
                try:
                    acc_b, res_b = eval_b.evaluate(test_ds, progress_callback=lambda p, m: update_progress(p,
                                                                                                           f"GPT-5 평가 중: {int(p * 100)}%"))
                except TypeError:
                    acc_b, res_b = eval_b.evaluate(test_ds)

                # 3. 로컬 평가
                status_text.write("🏠 로컬 모델 채점 시작...")
                eval_a = RFPEvaluator(st.session_state.gen_a)
                try:
                    acc_a, res_a = eval_a.evaluate(test_ds, progress_callback=lambda p, m: update_progress(p,
                                                                                                           f"로컬 모델 평가 중: {int(p * 100)}%"))
                except TypeError:
                    acc_a, res_a = eval_a.evaluate(test_ds)

                # 4. 결과 통합 (들여쓰기 수정됨: try-except 밖으로 이동하여 무조건 실행)
                status_text.write("📊 결과 집계 중...")
                comp = []
                for i, (b_res, a_res) in enumerate(zip(res_b, res_a)):
                    origin_file = test_ds.iloc[i].get('source', '파일정보없음')
                    comp.append({
                        "파일명": origin_file,
                        "질문": b_res.get('질문') or b_res.get('question'),
                        "정답": b_res.get('정답') or b_res.get('ground_truth'),
                        "GPT": b_res['AI 답변'],
                        "GPT결과": "✅" if b_res['결과'] == "정답" else "❌",
                        "로컬": a_res['AI 답변'],
                        "로컬결과": "✅" if a_res['결과'] == "정답" else "❌"
                    })

                st.session_state.update({
                    "comparison_results": comp,
                    "acc_b": acc_b,
                    "acc_a": acc_a
                })

                status_text.success("✅ 평가 완료! 리포트 탭을 확인하세요.")
                progress_bar.progress(100)
                time.sleep(1)
                st.balloons()
                st.rerun()  # 결과 반영을 위해 화면 새로고침

            except Exception as e:
                st.error(f"평가 중 오류 발생: {e}")
        else:
            st.error("⚠️ 두 엔진을 먼저 가동해주세요!")

# ==========================================
# [메인 화면 탭 구성]
# ==========================================
t1, t2, t3 = st.tabs(["💬 GPT-5 채팅", "🏠 로컬 채팅", "📊 성능 비교"])

with t1:
    if st.session_state.initialized_b:
        st.success("✅ GPT-5 준비 완료")
        for m in st.session_state.history_b:
            with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
                st.write(m.content)
        if p := st.chat_input("GPT에게 질문", key="chat_b"):
            st.session_state.history_b.append(HumanMessage(content=p))
            st.chat_message("user").write(p)
            ans = st.session_state.gen_b.generate_answer(p)
            st.session_state.history_b.append(AIMessage(content=ans))
            st.chat_message("assistant").write(ans)
    else:
        st.warning("GPT 엔진 미작동: 사이드바에서 상태를 확인하세요.")

with t2:
    if st.session_state.initialized_a:
        st.success("✅ 로컬 Llama3 준비 완료")
        for m in st.session_state.history_a:
            with st.chat_message("user" if isinstance(m, HumanMessage) else "assistant"):
                st.write(m.content)
        if p := st.chat_input("로컬 모델에게 질문", key="chat_a"):
            st.session_state.history_a.append(HumanMessage(content=p))
            st.chat_message("user").write(p)
            ans = st.session_state.gen_a.generate_answer(p)
            st.session_state.history_a.append(AIMessage(content=ans))
            st.chat_message("assistant").write(ans)
    else:
        st.warning("로컬 엔진 미작동: 사이드바에서 상태를 확인하세요.")

with t3:
    if st.session_state.comparison_results:
        st.subheader("🏁 모델 성능 비교 리포트")
        c1, c2 = st.columns([1, 2])  # 비율 조정 (그래프:표)

        with c1:
            # 막대 그래프 그리기
            fig, ax = plt.subplots(figsize=(4, 4))
            bars = ax.bar(['GPT', 'Local'], [st.session_state.acc_b, st.session_state.acc_a],
                          color=['#81B29A', '#E07A5F'])
            ax.set_ylim(0, 100)
            ax.set_ylabel("정확도 (%)")
            ax.set_title("모델 정확도 비교")

            # 막대 위에 점수 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}%', ha='center', va='bottom')

            st.pyplot(fig, use_container_width=True)

            st.metric("GPT 정확도", f"{st.session_state.acc_b}%")
            st.metric("로컬 정확도", f"{st.session_state.acc_a}%")

        with c2:
            st.write("### 📋 상세 결과표")
            st.dataframe(pd.DataFrame(st.session_state.comparison_results), use_container_width=True, height=500)
    else:
        st.info("👈 사이드바에서 '성능 비교 시작' 버튼을 눌러 평가를 진행하세요.")