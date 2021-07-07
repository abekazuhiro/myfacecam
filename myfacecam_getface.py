# coding: utf-8

import os
import cv2
import glob
import numpy as np

# カスケード分類器の設定
cascadefile = 'haarcascade_frontalface_default.xml'
assert os.path.isfile(cascadefile), cascadefile+'がみつかりません'

def main():
    global label

    # 学習データの設定
    label_len = 3
    label = str(input("ラベルの名前を英数３文字で指定してください(例:abe) > "))
    file_number = len(glob.glob(image_dir+"*"))

    # カメラ設定
    print("VideoCapture open")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Not Opened VideoCapture")
        exit()

    print("VideoCapture Start")
    while True:
        ret, img = cap.read()
        if not ret:
            print("VideoCapture Error")
            break

        # 顔を検出する
        img = getFace(img)
        # 保存した画像の枚数を表示する
        cv2.putText(img=img, text="{0}/{1}".format(str(face_count).zfill(3), max_photo_num),
            org=(80, 100), fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=1.5, color=(255,255,255), lineType=cv2.LINE_AA)

        # カメラ映像表示
        cv2.imshow("frame", img)
        # 10ミリ秒キーイベントを待って、何かキーが押されたら抜けます
        if cv2.waitKey(10) != -1:
            break

        # 指定された枚数画像を保存したら抜けます。
        if face_count > max_photo_num:
            break

    # 終了処理
    print("VideoCapture end")
    cap.release()
    cv2.destroyAllWindows()


def getFace(img):
    global cascade
    # 顔検出の効率化のためにモノクロにして情報量を落とす
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # 顔検出 minSize で顔判定する際の最小の四角の大きさを指定できる。(小さい値を指定し過ぎると顔っぽい小さなシミのような部分も判定されてしまう。)
    detectfaces = cascade.detectMultiScale(image=img_gray, scaleFactor=1.05, minNeighbors=10, minSize=(100,100))

    # 顔が検出できたら赤枠で表示する
    if len(detectfaces) > 0:
        for face in detectfaces:
        # こういう書き方もできる
        # for x, y, width, height in detectfaces:
            x = face[0]
            y = face[1]
            width = face[2]
            height = face[3]

            # 検出した顔を保存する
            face_img = img[y:y+height, x:x+width]
            saveFaceImg(face_img)

            # 顔を四角で囲む(thickness=太さ)
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+width, y+height), color=(0,0,255), thickness=3)

    return img

# １００ｘ１００の顔画像を保存する
def saveFaceImg(face_img):
    global face_count
    dst = cv2.resize(face_img, (100,100))
    filename = image_dir+label+"_{0}.jpg".format(face_count)
    cv2.imwrite(filename, dst)
    face_count += 1

if __name__ == '__main__':
    # カスケード分類器の定義
    cascade = cv2.CascadeClassifier(cascadefile)
    image_dir = "image/"
    face_count = 0
    label =""
    max_photo_num = 300
    main()
    print("end...")