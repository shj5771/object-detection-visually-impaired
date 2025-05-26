import os

# 세그먼테이션 데이터가 들어있는 폴더 경로
folder_path = r'C:\Users\User\Desktop\sub_street\dataset\labels\val'

# 삭제할 카테고리 번호
remove_category = 28

# 폴더 내 모든 .txt 파일에 대해 처리
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)

        # 파일 읽기
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # 카테고리 번호가 remove_category가 아닌 줄만 필터링
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            if int(parts[0]) != remove_category:
                filtered_lines.append(line)

        # 필터링된 줄을 다시 파일에 저장
        with open(file_path, 'w') as file:
            file.writelines(filtered_lines)

print(f"카테고리 {remove_category}가 제거된 파일이 저장되었습니다.")
