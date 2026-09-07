import cv2
import json
import numpy as np

# 保存した顔データを読み込む
try:
    with open("faces.json", "r") as f:
        saved_faces = json.load(f)
except FileNotFoundError:
    print("faces.json が見つかりません")
    exit()

if len(saved_faces) == 0:
    print("保存されている顔データがありません")
    exit()

# カメラ
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("カメラを起動できません")
    exit()

# 顔検出器
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

print("カメラを起動しました")
print("登録した顔をカメラに映してください")
print("Qキーで終了します")

while True:

    ret, frame = camera.read()

    if not ret:
        print("カメラ映像を取得できません")
        break

    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        # 顔画像を取得
        face_image = gray[y:y+h, x:x+w]

        # 100×100に統一
        face_image = cv2.resize(
            face_image,
            (100, 100)
        )

        # 数値データに変換
        current_feature = face_image.flatten().astype(np.float32)

        # 保存データと比較
        saved_feature = np.array(
            saved_faces[0]["feature"],
            dtype=np.float32
        )

        # 差の平均を計算
        difference = np.mean(
            np.abs(current_feature - saved_feature)
        )

        # 判定
        if difference < 50:
            name = saved_faces[0]["name"]
        else:
            name = "Unknown"

        # 顔を四角で囲む
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        # 名前を表示
        cv2.putText(
            frame,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 0, 0),
            2
        )

    cv2.imshow("Face Compare Test", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()