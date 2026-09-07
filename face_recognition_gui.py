import sys
import cv2
import json
import numpy as np

from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QInputDialog
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap


class FaceRecognitionWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition")
        self.resize(900, 750)

        # -------------------------
        # GUI
        # -------------------------

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("カメラを起動しています...")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.register_button = QPushButton("現在の顔を登録")
        self.register_button.clicked.connect(self.register_current_face)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.register_button)

        widget = QWidget()
        widget.setLayout(layout)

        self.setCentralWidget(widget)

        # -------------------------
        # 顔検出器
        # -------------------------

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # -------------------------
        # 保存データ
        # -------------------------

        self.faces_file = "faces.json"
        self.saved_faces = self.load_faces()

        # 現在検出している顔
        self.current_face_image = None

        # -------------------------
        # カメラ
        # -------------------------

        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            QMessageBox.critical(
                self,
                "エラー",
                "カメラを起動できませんでした。"
            )
            sys.exit()

        print("カメラOK")

        # -------------------------
        # タイマー
        # -------------------------

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    # ==================================================
    # faces.jsonを読み込む
    # ==================================================

    def load_faces(self):

        try:

            with open(self.faces_file, "r") as f:
                data = json.load(f)

                if isinstance(data, list):
                    return data

                return []

        except FileNotFoundError:

            return []

        except json.JSONDecodeError:

            return []

    # ==================================================
    # 顔データを保存
    # ==================================================

    def save_faces(self):

        with open(self.faces_file, "w") as f:

            json.dump(
                self.saved_faces,
                f,
                ensure_ascii=False,
                indent=2
            )

    # ==================================================
    # カメラ更新
    # ==================================================

    def update_camera(self):

        ret, frame = self.camera.read()

        if not ret:

            self.status_label.setText(
                "映像を取得できません"
            )

            return

        # 左右反転
        frame = cv2.flip(frame, 1)

        # グレースケール
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # 顔検出
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        self.current_face_image = None

        # 顔が見つかった場合
        if len(faces) > 0:

            # 最初の顔を使用
            x, y, w, h = faces[0]

            # 顔画像
            face_image = gray[y:y+h, x:x+w]

            # 100×100に統一
            face_image = cv2.resize(
                face_image,
                (100, 100)
            )

            self.current_face_image = face_image.copy()

            # 特徴データ
            current_feature = (
                face_image
                .flatten()
                .astype(np.float32)
            )

            # 初期状態
            name = "Unknown"
            best_difference = float("inf")

            # 保存データと比較
            for saved_face in self.saved_faces:

                saved_feature = np.array(
                    saved_face["feature"],
                    dtype=np.float32
                )

                difference = np.mean(
                    np.abs(
                        current_feature -
                        saved_feature
                    )
                )

                # 一番近い顔
                if difference < best_difference:

                    best_difference = difference
                    name = saved_face["name"]

            # 近さが50以上ならUnknown
            if (
                len(self.saved_faces) == 0
                or best_difference >= 50
            ):

                name = "Unknown"

            # 顔を四角で囲む
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

            # 名前
            cv2.putText(
                frame,
                name,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2
            )

            if name == "Unknown":

                self.status_label.setText(
                    "Unknown - 登録ボタンを押すと登録できます"
                )

            else:

                self.status_label.setText(
                    f"認識しました: {name}"
                )

        else:

            self.status_label.setText(
                "顔をカメラに映してください"
            )

        # -------------------------
        # GUIに映像表示
        # -------------------------

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        height, width, channel = frame.shape

        bytes_per_line = channel * width

        image = QImage(
            frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )

        pixmap = QPixmap.fromImage(image)

        self.label.setPixmap(
            pixmap.scaled(
                self.label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ==================================================
    # 顔を登録
    # ==================================================

    def register_current_face(self):

        # 顔がない場合
        if self.current_face_image is None:

            QMessageBox.warning(
                self,
                "顔がありません",
                "まず顔をカメラに映してください。"
            )

            return

        # 登録確認
        result = QMessageBox.question(
            self,
            "顔を登録",
            "この顔を登録しますか？",
            QMessageBox.Yes | QMessageBox.No
        )

        if result != QMessageBox.Yes:

            return

        # 名前入力
        name, ok = QInputDialog.getText(
            self,
            "名前を登録",
            "名前を入力してください:"
        )

        if not ok or not name.strip():

            return

        name = name.strip()

        # 特徴データ作成
        feature_data = (
            self.current_face_image
            .flatten()
            .tolist()
        )

        # データ追加
        self.saved_faces.append(
            {
                "name": name,
                "feature": feature_data
            }
        )

        # 保存
        self.save_faces()

        QMessageBox.information(
            self,
            "登録完了",
            f"{name} さんを登録しました！"
        )

        self.status_label.setText(
            f"登録完了: {name}"
        )

    # ==================================================
    # 終了
    # ==================================================

    def closeEvent(self, event):

        self.camera.release()

        event.accept()


# ======================================================
# アプリ起動
# ======================================================

app = QApplication(sys.argv)

window = FaceRecognitionWindow()

window.show()

sys.exit(app.exec())