# @title src/analysis_metrics.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 데이터 설정
labels = np.array(['정확도', '검색 정밀도', '환각 방지', '확장자 인식', '답변 관련성'])
stats = np.array([95, 90, 98, 100, 92]) # 최종 모델 수치(예시)

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
stats = np.append(stats, stats[0])
angles = np.append(angles, angles[0])

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
# 진한 파스텔 초록색 적용
ax.fill(angles, stats, color='#81B29A', alpha=0.5)
ax.plot(angles, stats, color='#81B29A', linewidth=2)

ax.set_yticklabels([])
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
plt.title('최종 RAG 시스템 성능 지표 평가', size=15, pad=20)
plt.show()