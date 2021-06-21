# coding: utf-8

#test
import cv2
import numpy as np

def main():
    # 変数定義

    # カメラ設定
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # H264にしてカメラ読み込み時間を高速化
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'));

    if not cap.isOpened():
        print("Not Opened VideoCapture.")
        exit()

    while True:
        # カメラ映像読み込み
        ret, img = cap.read()
        if not ret:
            print("VideoCapture Error.")
            break

        # カメラ映像表示
        cv2.imshow("frame", img)
        if cv2.waitKey(10) != -1:
            break

    # 終了処理
    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    main()
