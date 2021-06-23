# coding: utf-8

import cv2
import os
import numpy as np

assert os.path.isfile('haarcascade_frontalface_default.xml'), 'haarcascade_frontalface_default.xml がない'

def main():
    # 変数定義

    # カメラ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # H264にしてカメラ読み込み時間を高速化
#    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));

    if not cap.isOpened():
        print("Not Opened VideoCapture.")
        exit()

    while True:
        # カメラ映像読み込み
        ret, img = cap.read()
        if not ret:
            print("VideoCapture Error.")
            break

        # カメラから読み込んだ画像から顔を検出する
        img = getFace(img)

        # カメラ映像表示
        cv2.imshow("frame", img)
        if cv2.waitKey(10) != -1:
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()

# 顔検出関数
# 顔を検出して顔を四角で囲む
def getFace(img):
    global cascade

    # 顔検出の効率化のためにモノクロにして情報量を落とす
    gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # 顔検出
    detectrects = cascade.detectMultiScale(image=gray, scaleFactor=1.05, minNeighbors=10, flags=None, minSize=(30,30))
    # 顔が検出されたら赤枠を描画する
    if len(detectrects) > 0:
        for rect in detectrects:
            cv2.rectangle(img=img, pt1=tuple(rect[0:2]), pt2=tuple(rect[0:2]+rect[2:4]), color=(0, 0, 255), thickness=3)
    return img

if __name__ == '__main__':
    # カスケード分類器の定義
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    main()
