import sys
import cv2

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap


class CameraWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Recognition")
        self.resize(900, 700)

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # カメラ
        self.camera = cv2.VideoCapture(0)

        if not self.camera.isOpened():
            print("カメラを起動できませんでした")
            sys.exit()

        print("カメラOK")

        # 顔検出器
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        # カメラ更新
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

    def update_camera(self):

        ret, frame = self.camera.read()

        if not ret:
            print("映像を取得できません")
            return

        # 左右反転
        frame = cv2.flip(frame, 1)

        # グレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔を検出
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        # 顔を四角で囲む
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

        # RGBに変換
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

    def closeEvent(self, event):
        self.camera.release()
        event.accept()


app = QApplication(sys.argv)

window = CameraWindow()
window.show()

sys.exit(app.exec())