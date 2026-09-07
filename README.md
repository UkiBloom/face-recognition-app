# Face Recognition App

Python と OpenCV を使用した、カメラ映像から顔を検出・認識するデスクトップアプリです。
大学時代にJavaで顔認証システムを作成したことがあり、大学時代から現在まで約10年間のブランクがあることから、感覚を取り戻すため作成しました。

カメラに映った顔と、あらかじめ登録した顔データを比較し、登録済みの人物であれば名前を表示します。

## Features

* 📷 カメラ映像のリアルタイム表示
* 👤 顔の検出
* 🔍 登録済みの顔との比較
* 🏷️ 認識した人物の名前を表示
* ❓ 未登録の人物は `Unknown` と表示
* 💾 顔データの登録・保存
* 🖥️ PySide6 によるGUI

## Demo

カメラに顔を映すと、顔を検出して登録済みの人物かどうかを判定します。

認識した場合：

```text
田中さん
```

登録されていない場合：

```text
Unknown
```

## Technologies

* Python 3
* OpenCV
* PySide6
* JSON

## Project Structure

```text
face-recognition-app/
├── camera_gui.py
├── face_detect.py
├── face_feature.py
├── face_feature_test.py
├── face_compare_test.py
├── face_recognition_app.py
├── face_recognition_gui.py
├── face_sample.jpg
├── .gitignore
└── README.md
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/UkiBloom/face-recognition-app.git
cd face-recognition-app
```

### 2. Install dependencies

```bash
pip3 install opencv-contrib-python PySide6 Pillow
```

### 3. Run the application

```bash
python3 face_recognition_gui.py
```

## How It Works

1. カメラから映像を取得
2. OpenCV の Haar Cascade を使用して顔を検出
3. 検出した顔画像をグレースケール化・リサイズ
4. 登録済みの顔データと比較
5. 一定以上一致した場合、登録された名前を表示
6. 一致しない場合は `Unknown` と表示
7. 必要に応じて新しい顔を登録

## Data Privacy

顔データは個人情報を含む可能性があるため、登録された顔データ `faces.json` はGitHubリポジトリには含めていません。

`.gitignore` により、ローカル環境の顔データが誤ってGitHubへアップロードされることを防止しています。

## Notes

現在の顔認識処理は、顔画像をグレースケール化してリサイズした画像同士を比較するシンプルな方式です。

そのため、照明・顔の角度・表情などの影響を受ける可能性があります。

今後は、より精度の高い顔特徴量（Face Embedding）を利用した認識方式への改善を予定しています。

## Future Improvements

* [ ] 未登録の顔を検出した際の自動登録フロー
* [ ] 顔認識精度の向上
* [ ] Face Embedding を利用した認識
* [ ] 登録済みユーザーの削除機能
* [ ] 登録ユーザー一覧の表示
* [ ] Mac向けアプリ（`.app`）としてパッケージ化
* [ ] UIデザインの改善

## Author

UkiBloom

---

This project was created as a Python/OpenCV learning and portfolio project.
