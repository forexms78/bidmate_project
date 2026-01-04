네, 프로젝트의 핵심 기능(이원화된 RAG, 평가 시스템, 대시보드)이 잘 드러나면서도, **요즘 깃허브 트렌드에 맞는 깔끔하고 세련된 스타일**로 `README.md`를 작성해 드립니다.

아래 내용을 복사해서 `README.md` 파일에 붙여넣으세요. (괄호로 표시된 부분만 본인 상황에 맞게 조금 수정하시면 됩니다.)

---

## 📋 README.md 소스 코드

```markdown
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

> *(여기에 실행 화면 스크린샷이나 GIF를 넣어주세요. 예: `assets/demo.gif`)*
> ![Dashboard Screenshot](./assets/dashboard_screenshot.png)

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

```mermaid
graph LR
    A[RFP Documents] --> B(RFP Data Loader)
    B --> C{Vector DB Builder}
    C -->|API Mode| D[ChromaDB (Main)]
    C -->|Local Mode| E[ChromaDB (Local)]
    D --> F[GPT Generator]
    E --> G[Local Generator]
    F & G --> H[Streamlit Dashboard]
    H --> I[Performance Evaluator]

```

## 🚀 Quick Start

### 1. Installation

프로젝트를 클론하고 필수 패키지를 설치합니다.

```bash
git clone [https://github.com/your-username/bidmate-rag.git](https://github.com/your-username/bidmate-rag.git)
cd bidmate-rag
pip install -r requirements.txt

```

### 2. Configuration

`.env` 파일을 생성하고 API 키를 설정합니다.

```bash
OPENAI_API_KEY=sk-proj-...

```

### 3. Run Application

Streamlit 대시보드를 실행합니다.

```bash
streamlit run app.py

```

## 📂 Directory Structure

```bash
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

```

## 📝 License

This project is licensed under the **MIT License**.

---

<div align="center">
Developed by <b>BidMate Team</b> | Powered by LangChain & Streamlit
</div>

```

-----

### 💡 더 멋있게 만드는 꿀팁 (이건 꼭 하세요\!)

1.  **스크린샷 추가 (필수):**
      * `assets`라는 폴더를 만들고, 앱 실행 화면을 캡처해서 `dashboard_screenshot.png`라는 이름으로 저장해 넣으세요.
      * 글자보다 **사진 한 장**이 프로젝트를 10배 더 있어 보이게 만듭니다.
2.  **배지(Badges) 활용:**
      * 제가 넣어드린 `shields.io` 배지는 깃허브에서 아주 예쁘게 렌더링 됩니다. 기술 스택을 한눈에 보여줍니다.
3.  **Mermaid 차트:**
      * `System Architecture` 부분에 제가 넣어드린 코드는 깃허브에서 \*\*자동으로 다이어그램(순서도)\*\*으로 변환되어 보입니다. 아주 전문적으로 보일 겁니다.

이대로 올리시면 포트폴리오로 쓰기에도 손색없을 겁니다\!

```
