# coding: utf-8

import os
import cv2
import numpy as np
from PIL import Image
import pickle
from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

images = []
labels = []

def main():
    global labels
    global images
    print("Create Machine Learning Model start")
    for imagefile in os.listdir(image_dir):
        #画像のパス取得
        image_path = os.path.join(image_dir, imagefile)
        if image_path.endswith('.jpg'):
            # 画像をグレースケールで読み込み、画像データ、ラベルを取得する
            gray_image = Image.open(image_path).convert("L")
            # nummpy配列に格納
            image = np.array(gray_image, "uint8")
            # imageを1次元配列に変換
            image = image.flatten()
            images.append(image)
            # ファイル名からラベル名取得
            labels.append(str(imagefile[0:3]))

    # 行列に変換
    labels = np.array(labels)
    images = np.array(images)

    # svmの変換器を作成(clf: classfication 分類の意)
    print("create: "+filename_svm)
    clf = svm.LinearSVC()
    clf.fit(images, labels)
    pickle.dump(clf, open(filename_svm,"wb"))

    # KNNの変換器を作成(K-Nearest Neighbor)
    print("create: "+filename_knn)
    knn = KNeighborsClassifier(n_neighbors=k_value)
    knn.fit(images, labels)
    pickle.dump(knn, open(filename_knn, "wb"))

    print("モデル生成完了")


if __name__ == '__main__':
    image_dir = 'image'
    filename_svm = "face_model_svm.sav"
    k_value = int(input("K値を指定してください > "))
#   k_value = 5
    filename_knn = "face_model_knn_{0}.sav".format(k_value)
    main()
