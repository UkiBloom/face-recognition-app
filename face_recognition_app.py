import cv2
import numpy as np
import os
import json

# =========================
# 設定
# =========================

DATA_FILE = "faces.json"
FACE_SIZE = (200, 200)

# =========================
# 登録データを読み込む
# =========================

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        registered_faces = json.load(f)
else:
    registered_faces = {}


# =========================
# 顔検出モデル
# =========================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# =========================
# カメラ
# =========================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("カメラを起動できませんでした")
    exit()

print("カメラを起動しました")
print("顔を映してください")
print("qキーで終了します")

# =========================
# 登録画面
# =========================

def register_face(face):

    print()
    print("======================")
    print("新しい顔を検出しました")
    print("この人を登録しますか？")
    print("======================")

    answer = input("登録する？ (y/n): ")

    if answer.lower() != "y":
        print("登録しませんでした")
        return

    name = input("名前を入力してください: ")

    if name.strip() == "":
        print("名前が入力されていません")
        return

    # 顔データを保存
    face = cv2.resize(face, FACE_SIZE)

    face_list = face.tolist()

    registered_faces[name] = face_list

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            registered_faces,
            f,
            ensure_ascii=False
        )

    print()
    print("登録しました！")
    print("名前:", name)
    print("======================")


# =========================
# 顔認識
# =========================

while True:

    ret, frame = camera.read()

    if not ret:
        print("映像を取得できません")
        break

    # グレースケール
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # 顔を検出
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    # =========================
    # 顔をチェック
    # =========================

    for (x, y, w, h) in faces:

        face = gray[
            y:y + h,
            x:x + w
        ]

        face = cv2.resize(
            face,
            FACE_SIZE
        )

        recognized_name = "Unknown"
        lowest_difference = float("inf")

        # =========================
        # 登録されている顔と比較
        # =========================

        for name, saved_face in registered_faces.items():

            saved_face = np.array(
                saved_face,
                dtype=np.uint8
            )

            difference = np.mean(
                cv2.absdiff(
                    saved_face,
                    face
                )
            )

            if difference < lowest_difference:
                lowest_difference = difference
                recognized_name = name

        # =========================
        # 類似度判定
        # =========================

        threshold = 45

        if lowest_difference >= threshold:
            recognized_name = "Unknown"

        # =========================
        # 画面表示
        # =========================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            recognized_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        # =========================
        # Unknownなら登録
        # =========================

        if recognized_name == "Unknown":

            cv2.putText(
                frame,
                "Press R to register",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # =========================
    # カメラ画面
    # =========================

    cv2.imshow(
        "Face Recognition",
        frame
    )

    # キー入力
    key = cv2.waitKey(1) & 0xFF

    # q → 終了
    if key == ord("q"):
        break

    # r → Unknownを登録
    if key == ord("r") and len(faces) == 1:

        x, y, w, h = faces[0]

        face = gray[
            y:y + h,
            x:x + w
        ]

        register_face(face)


# =========================
# 終了
# =========================

camera.release()
cv2.destroyAllWindows()

print("終了しました")