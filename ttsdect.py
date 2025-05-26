import cv2
import random
import numpy as np
import time
import os
import threading
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
from gtts import gTTS

# ========== 설정 ==========
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_SIZE = 24
VIDEO_PATH = r"C:/Users/User/Desktop/tts/testvid.mp4"
OUTPUT_PATH = r"C:/Users/User/Desktop/tts/testvid_result2.mp4"
MODEL_PATH = r"C:/Users/User/Desktop/tts/best.pt"
FFMPEG_PATH = r"C:/Users/User/Downloads/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"
TEMP_TTS_DIR = os.path.join(os.getcwd(), "temp_tts")
os.makedirs(TEMP_TTS_DIR, exist_ok=True)

# TTS 최소 간격 설정 (초 단위)
TTS_MIN_INTERVAL = 2.0

# ========== 초기화 ==========
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# 클래스 이름 및 우선순위
id2name = {0: "barricade", 1: "bench", 2: "bicycle", 3: "bollard", 4: "bus", 5: "car", 6: "carrier", 7: "cat",
           8: "chair", 9: "dog", 10: "fire_hydrant", 11: "kiosk", 12: "motorcycle", 13: "movable_signage",
           14: "parking_meter", 15: "person", 16: "pole", 17: "potted_plant", 18: "power_controller",
           19: "scooter", 20: "stop", 21: "stroller", 22: "table", 23: "traffic_light", 24: "traffic_light_controller",
           25: "traffic_sign", 26: "tree_trunk", 27: "truck", 28: "wheelchair", 29: "off", 30: "green_traffic_light",
           31: "red_traffic_light", 32: "braille_block", 33: "crosswalk", 34: "kick"}
priority = {"crosswalk": 0.2, "green_traffic_light": 0.7, "red_traffic_light": 1.0}
alert_kor = {"green_traffic_light": "초록불입니다 이동하세요.", "red_traffic_light": "신호등이 빨간불입니다.",
            "crosswalk": "횡단보도 발견", "bollard": "볼라드 발견", "person": "사람입니다 조심하세요."}
name2color = {name: tuple(random.randint(50, 255) for _ in range(3)) for name in id2name.values()}

# 상태 변수
tts_schedule = []
frame_count = 0
frame_time = lambda fc: fc / fps
green_light_last_time = -1000
car_alert_schedule = []
CAR_ALERT_DURATION = 1.0
alert_last_time = {}
ALERT_COOLDOWN = 30.0

# TTS 추가 간격 검사 함수
def can_add_tts(current_sec):
    return all(abs(current_sec - prev_sec) > TTS_MIN_INTERVAL for prev_sec, *_ in tts_schedule)

# 텍스트 배경 그리기 함수
def draw_text_with_background(base_image, text, position, font, text_color=(0, 0, 0), bg_color=(255, 255, 255, 200), radius=8):
    draw = ImageDraw.Draw(base_image)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = position
    padding = 10
    box_x2, box_y2 = x + text_width + padding, y + text_height + padding
    ascent, descent = font.getmetrics()
    visual_height = ascent + descent
    y_offset = (text_height - visual_height) // 2
    text_x = x + (box_x2 - x - text_width) // 2
    text_y = y + (box_y2 - y - visual_height - 10) // 2 - y_offset
    temp = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp)
    temp_draw.rounded_rectangle([x, y, box_x2, box_y2], radius=radius, fill=bg_color)
    temp_draw.text((text_x, text_y), text, font=font, fill=text_color)
    base_image.alpha_composite(temp)

# 영상 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    current_sec = frame_time(frame_count)
    results = model(frame, conf=0.2)

    # 우선순위 계산 변수
    highest_priority = None
    highest_score = -1

    # 탐지 및 TTS 예약
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in id2name:
                continue
            name = id2name[cls_id]

            # 녹색 불 업데이트
            if name == "green_traffic_light":
                green_light_last_time = current_sec
            # 바로 직전 녹색 불 후 빨강 불 무시
            if name == "red_traffic_light" and (current_sec - green_light_last_time <= 3.0):
                continue

            # 경고음성 예약
            if name in alert_kor and (name not in alert_last_time or current_sec - alert_last_time[name] >= ALERT_COOLDOWN):
                if can_add_tts(current_sec):
                    msg = alert_kor[name]
                    tts_schedule.append((current_sec, name, msg))
                    alert_last_time[name] = current_sec

            # 자동차 경고 예약 (쿨다운 없이)
            if name in ["car", "truck"]:
                if can_add_tts(current_sec):
                    car_alert_schedule.append(current_sec)
                    tts_schedule.append((current_sec, "car_warning", "자동차 조심하세요"))

            # 우선순위 계산 (화면 표시용)
            if name in priority:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                dist_score = area / (width * height)
                total_score = priority[name] + dist_score
                if total_score > highest_score:
                    highest_score = total_score
                    highest_priority = {"name": name, "box": (x1, y1, x2, y2)}

    # 화면에 가장 높은 우선순위 텍스트 표시
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    if highest_priority:
        msg = alert_kor.get(highest_priority["name"], None)
        if msg:
            draw_text_with_background(frame_pil, msg, (30, 30), font)

    # 자동차 경고 텍스트 표시
    if any((current_sec - t <= CAR_ALERT_DURATION) for t in car_alert_schedule):
        draw_text_with_background(frame_pil, "자동차 조심!!!", (width // 2 - 100, height // 2 - 100), font)

    # 프레임 변환
    frame = cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2BGR)

    # 박스 그리기
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in id2name:
                continue
            name = id2name[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            is_highest = highest_priority and (x1, y1, x2, y2) == highest_priority["box"]
            color = (0, 0, 255) if is_highest else name2color.get(name, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# ========== TTS 개별 생성 및 타이밍 조정 ==========
if tts_schedule:
    segments = []
    for idx, (sec, key, text) in enumerate(tts_schedule):
        try:
            tts = gTTS(text=text, lang='ko')
            tts_path = os.path.join(TEMP_TTS_DIR, f"tts_{idx}_{key}.mp3")
            tts.save(tts_path)
            segments.append((sec, tts_path))
        except Exception as e:
            print(f"TTS 생성 실패 ({key}): {e}")

    filter_complex = ""
    inputs = []
    amix_parts = []

    for i, (delay_sec, path) in enumerate(segments):
        delay_ms = int(delay_sec * 1000)
        delay_str = f"adelay={delay_ms}|{delay_ms}";
        inputs.append(f"-i \"{path}\"")
        filter_complex += f"[{i+1}:a]{delay_str}[a{i}];"
        amix_parts.append(f"[a{i}]")

    amix_str = f"{''.join(amix_parts)}amix=inputs={len(amix_parts)}[aout]"
    filter_complex += amix_str

    final_output = OUTPUT_PATH.replace(".mp4", "_with_audio.mp4")
    cmd = (
        f"{FFMPEG_PATH} -y -i \"{OUTPUT_PATH}\" {' '.join(inputs)} "
        f"-filter_complex \"{filter_complex}\" -map 0:v -map \"[aout]\" -c:v copy -c:a aac -shortest \"{final_output}\""
    )
    print("[FFMPEG 명령 실행]:", cmd)
    os.system(cmd)
else:
    print("⚠️ TTS 없음: 영상만 저장됨 ->", OUTPUT_PATH)