import os
import pandas as pd
import glob

# 경로 설정
BASE_DIR = "DATA"
FILES_DIR = os.path.join(BASE_DIR, "files")
CSV_PATH = os.path.join(BASE_DIR, "data_list.csv")


def check_data_structure():
    print("=== 1. 폴더 구조 확인 ===")
    if not os.path.exists(FILES_DIR):
        print(f"❌ '{FILES_DIR}' 폴더가 없습니다. 경로를 확인해주세요.")
        return

    # 실제 존재하는 HWP 파일 리스트 확인
    real_files = os.listdir(FILES_DIR)
    hwp_files = [f for f in real_files if f.endswith('.hwp')]
    print(f"📂 'files' 폴더 내 파일 개수: {len(real_files)}개")
    print(f"📄 그 중 HWP 파일 개수: {len(hwp_files)}개")
    print(f"👀 파일명 예시 (상위 3개):\n {real_files[:3]}\n")

    print("=== 2. 메타데이터(CSV) 확인 ===")
    if os.path.exists(CSV_PATH):
        try:
            # 인코딩 문제 방지를 위해 utf-8 또는 cp949 시도
            try:
                df = pd.read_csv(CSV_PATH, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(CSV_PATH, encoding='cp949')

            print(f"✅ CSV 로드 성공! (행: {df.shape[0]}, 열: {df.shape[1]})")
            print(f"📋 컬럼 목록: {list(df.columns)}")
            print("📊 데이터 예시 (상위 1행):")
            print(df.head(1).T)  # 보기 편하게 전치

            # 파일명 매칭 테스트 (파일명 컬럼이 있다고 가정하고 추측)
            # 보통 '파일명', 'file_name', '공고명' 등이 파일명과 연관됨
            print("\n=== 3. 파일명 매칭 테스트 ===")
            # 파일명과 관련된 컬럼이 있는지 확인해봅니다.
            potential_cols = [col for col in df.columns if '파일' in col or 'File' in col or '제목' in col or '공고명' in col]
            print(f"매칭 후보 컬럼: {potential_cols}")

        except Exception as e:
            print(f"❌ CSV 읽기 실패: {e}")
    else:
        print(f"❌ '{CSV_PATH}' 파일이 없습니다.")

    print("\n=== 4. 엑셀 파일 확인 (추가) ===")
    # xlsx 파일이 있다면 하나만 열어서 구조 확인
    xlsx_files = glob.glob(os.path.join(BASE_DIR, "*.xlsx"))
    if xlsx_files:
        print(f"엑셀 파일 발견: {xlsx_files}")
        try:
            df_xl = pd.read_excel(xlsx_files[0])
            print(f"📋 엑셀({os.path.basename(xlsx_files[0])}) 컬럼 목록: {list(df_xl.columns)}")
        except Exception as e:
            print(f"❌ 엑셀 읽기 실패: {e}")
    else:
        print("엑셀 파일이 없습니다.")


if __name__ == "__main__":
    check_data_structure()