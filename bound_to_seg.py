import os


def convert_line(line):
    class_id, x_center, y_center, width, height = map(float, line.strip().split())

    # Bounding box to polygon (clockwise)
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center - height / 2
    x3 = x_center + width / 2
    y3 = y_center + height / 2
    x4 = x_center - width / 2
    y4 = y_center + height / 2

    return f"{int(class_id)} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"


def convert_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            txt_path = os.path.join(folder_path, filename)
            with open(txt_path, "r") as f:
                lines = f.readlines()

            # 바운딩박스 라인을 세그멘테이션 라인으로 변환
            converted_lines = [convert_line(line) for line in lines]

            # 덮어쓰기
            with open(txt_path, "w") as f:
                f.write("\n".join(converted_lines))


# 여기에 당신의 실제 폴더 경로 입력
convert_folder(r"C:\Users\User\Desktop\traffic\traffic_light\labels")
