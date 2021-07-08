# coding: utf-8

import os
import cv2
import numpy as np
import pickle

# カスケード分類器の設定
cascadefile = 'haarcascade_frontalface_default.xml'
assert os.path.isfile(cascadefile), cascadefile+'がみつかりません'

def main():
    print("myface recognizer start")

    # カメラ設定
    print("VideoCapture open")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Not Opened VideoCapture")
        exit()

    # キャンセルされるまで繰り返し
    print("VideoCapture start")
    while True:
        ret, img = cap.read()
        if not ret:
            print("VideoCapture Error")
            break

        # 顔を検出して判定する
        faceRecognize(img)
        
        # カメラ映像表示
        cv2.imshow("frame", img)

        # 何かキーが押されたら抜けます
        if cv2.waitKey(10) != -1:
            break

    # 終了処理
    print("Videocapture end")
    cap.release()
    cv2.destroyAllWindows()

def faceRecognize(img):
    global cascade
    # 顔検出の効率化のためにモノクロにして情報量を落とす
    img_gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    # 顔検出 
    detectfaces = cascade.detectMultiScale(image=img_gray, scaleFactor=1.05, minNeighbors=10, minSize=(10,10))

    # 顔が見つかったら判定する
    if(len(detectfaces)) > 0:
        for face in detectfaces:
            x = face[0]
            y = face[1]
            width = face[2]
            height = face[3]

            # 顔を四角で囲む(thickness=太さ)
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x+width, y+height), color=(0,0,255), thickness=3)

            # 検出した顔を切り取る
            face_img = img_gray[y:y+height, x:x+width]
            # 学習データを作った画像に合わせて100x100にリサイズ
            recog_img = cv2.resize(face_img, (100,100), interpolation=cv2.INTER_LINEAR)
            # 1次元の行列に変換
            recog_array = np.array(recog_img, "uint8").flatten()
            
            # 検出した顔を判定する
            # predict()は配列ではなく行列を引数とするので、行列に変換する
            # Expected 2D array, got 1D array instead:...
            # 行列を用意して
            recog_matrix = []
            # 配列を要素として追加して（このままだと１次元配列）
            recog_matrix.append(recog_array)
            # 行列に変換する
            recog_matrix = np.array(recog_matrix)
            # 予測実行
            pred = clf.predict(recog_matrix)

            # 予測の結果から名前を表示する
            cv2.putText(img=img, text=str(pred[0]), org=(x+int(width*1/3), y-int(height/8)), 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=width/150, 
                color=(255,255,255), thickness=int(width/100),
                lineType=cv2.LINE_AA)


if __name__ == '__main__':
    cascade = cv2.CascadeClassifier(cascadefile)
    clf = pickle.load(open("face_model_svm.sav","rb"))
    main()