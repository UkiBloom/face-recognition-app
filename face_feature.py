import cv2
import os

# カメラを起動
camera = cv2.VideoCapture(0)

print("カメラを起動しました")
print("顔を映してください")
print("sキーで顔を保存します")
print("qキーで終了します")

# 顔検出モデル
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

while True:

    # カメラから映像を取得
    ret, frame = camera.read()

    if not ret:
        print("映像を取得できませんでした")
        break

    # グレースケールに変換
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

    # 検出した顔を処理
    for (x, y, w, h) in faces:

        # 顔を青い四角で囲む
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    # カメラ映像を表示
    cv2.imshow(
        "Face Feature",
        frame
    )

    # キーボード入力
    key = cv2.waitKey(1) & 0xFF

    # sキー → 顔を保存
    if key == ord("s"):

        # 顔が1つだけの場合
        if len(faces) == 1:

            x, y, w, h = faces[0]

            # 顔部分を切り出す
            face = gray[
                y:y + h,
                x:x + w
            ]

            # 顔のサイズを統一
            face = cv2.resize(
                face,
                (200, 200)
            )

            # 顔画像を保存
            cv2.imwrite(
                "face_sample.jpg",
                face
            )

            print("顔のデータを保存しました！")
            print("保存先：face_sample.jpg")

        elif len(faces) == 0:

            print("顔が見つかりません")

        else:

            print("複数の顔が映っています")
            print("1人だけカメラに映してください")

    # qキー → 終了
    if key == ord("q"):
        break

# カメラを終了
camera.release()

# ウィンドウを閉じる
cv2.destroyAllWindows()

print("終了しました")