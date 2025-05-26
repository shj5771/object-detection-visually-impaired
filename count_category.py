import os
from collections import defaultdict


def count_categories_in_txt_files(folder_path):
    category_counts = defaultdict(int)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            new_lines = []  # 수정된 데이터를 저장할 리스트

            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()

                    try:
                        # 첫 번째 값이 카테고리인 경우
                        category = int(parts[0])
                        category_counts[category] += 1
                        new_lines.append(line)  # 카테고리 값을 제대로 읽은 경우에는 그대로 저장
                    except ValueError:
                        # 첫 번째 값이 실수인 경우 카테고리 번호가 아니므로 해당 열을 삭제
                        try:
                            float(parts[0])  # 첫 번째 값이 실수이면 좌표값임
                            parts.pop(0)  # 첫 번째 열(좌표값)을 삭제
                            new_line = ' '.join(parts)  # 나머지 데이터만 저장
                            new_lines.append(new_line)  # 수정된 라인 저장
                        except ValueError:
                            print(f"Skipping malformed line in {filename}: {line}")

            # 수정된 데이터를 파일에 다시 저장
            with open(file_path, 'w') as file:
                file.write('\n'.join(new_lines) + '\n')

    # 정렬해서 출력
    total = 0
    for category in sorted(category_counts):
        total += category_counts[category]
        print(f"Category {category}: {category_counts[category]} occurrences")


# 사용 예:
folder_path = r'C:\Users\User\Desktop\15000\labels'  # 여기에 txt 파일들이 있는 폴더 경로를 입력하세요
count_categories_in_txt_files(folder_path)
