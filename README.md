# 🏢 입찰메이트 (BidMate) - Integrated RAG Dashboard

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
  ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white)
  ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white)
  ![ChromaDB](https://img.shields.io/badge/ChromaDB-CC5500?style=for-the-badge&logoColor=white)

  <br>
  
  **공공 입찰 제안요청서(RFP) 분석 및 예산/과업 정보 추출을 위한 하이브리드 RAG 솔루션**
  <br>
  GPT API와 로컬 LLM의 성능을 실시간으로 비교하고 평가합니다.

</div>

---

## 📸 Dashboard Preview

<img width="1914" height="904" alt="스크린샷 2026-01-04 190958" src="https://github.com/user-attachments/assets/437dfb77-ff22-4871-b72b-d528c858be05" />

<br>

## ✨ Key Features

### 1. 🧠 Hybrid RAG Engine (Dual System)
- **API Mode:** OpenAI GPT-4o/Mini를 활용한 고정확도 분석 및 요약.
- **Local Mode:** 보안이 중요한 환경을 위한 온프레미스(On-Premise) LLM 구동.
- 두 엔진의 답변을 동시에 비교하여 최적의 모델을 선택할 수 있습니다.

### 2. 📑 Advanced Data Loading
- **HWP/PDF 완벽 지원:** 공공기관 필수 포맷인 한글(.hwp) 파일의 텍스트 및 메타데이터 정밀 추출.
- **Smart Metadata Filter:** 파일 확장자(.hwp, .pdf)를 메타데이터로 태깅하여 문서 유형별 정확한 필터링 검색 지원.
- **AI Summary:** 긴 제안요청서를 LLM이 사전 요약하여 검색 정확도(Retriever) 향상.

### 3. 📊 Auto-Evaluation & Visualization
- **Ground Truth 자동 생성:** 원본 데이터 기반으로 예산, 발주기관, 과업범위 정답셋 자동 구축.
- **Real-time Scoring:** GPT와 로컬 모델의 정답률을 실시간으로 채점.
- **Interactive Dashboard:** Streamlit 기반의 직관적인 비교 차트 및 상세 결과 테이블 제공.

<br>

## 🛠️ System Architecture

🚀 Quick Start
1. Installation
프로젝트를 클론하고 필수 패키지를 설치합니다.

Bash

git clone [https://github.com/your-username/bidmate-rag.git](https://github.com/your-username/bidmate-rag.git)
cd bidmate-rag
pip install -r requirements.txt
2. Configuration
.env 파일을 생성하고 API 키를 설정합니다.

Bash

OPENAI_API_KEY=sk-proj-...
3. Run Application
Streamlit 대시보드를 실행합니다.

Bash

streamlit run app.py
📂 Directory Structure
Bash

├── app.py                  # 메인 대시보드 실행 파일
├── src/
│   ├── data_loader.py      # HWP/PDF 로더 및 메타데이터 처리
│   ├── vector_db.py        # ChromaDB 구축 및 관리
│   ├── generator.py        # LLM 답변 생성 로직
│   ├── evaluation.py       # 정답 채점 및 평가 모듈
│   └── evaluation_dataset_builder.py # 평가 데이터셋 생성기
├── local_src/              # 로컬 LLM 관련 모듈
├── DATA/                   # 제안요청서 원본 데이터
└── requirements.txt        # 의존성 패키지 목록
📝 License
This project is licensed under the MIT License.

<div align="center"> Developed by <b>BidMate Team</b> | Powered by LangChain & Streamlit </div>

협업일지
[
](https://www.notion.so/2de5df876c2080b8a979dc1cbbbbcc2a?source=copy_link)
