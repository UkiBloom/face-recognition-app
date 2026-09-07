import cv2
import json
import os

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
print("顔をカメラに映してください")
print("顔が検出されたら S キーで保存します")
print("終了する場合は Q キー")

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

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    cv2.imshow("Face Feature Test", frame)

    key = cv2.waitKey(1) & 0xFF

    # Sキーで保存
    if key == ord("s"):

        if len(faces) == 0:
            print("顔が検出されていません")
            continue

        # 最初に検出した顔を使用
        x, y, w, h = faces[0]

        face_image = gray[y:y+h, x:x+w]

        # 顔画像を小さなサイズに統一
        face_image = cv2.resize(
            face_image,
            (100, 100)
        )

        # 数値データに変換
        feature_data = face_image.flatten().tolist()

        # 保存するデータ
        data = {
            "name": "test",
            "feature": feature_data
        }

        with open("faces.json", "w") as f:
            json.dump([data], f)

        print("顔の特徴データを保存しました")
        print("データ数:", len(feature_data))

    # Qキーで終了
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()