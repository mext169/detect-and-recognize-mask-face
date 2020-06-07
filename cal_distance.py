# 计算输入图片和数据库中图片的距离
import os
import cv2
import face2
import numpy as np
import matplotlib.pyplot as plt


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.ifwear = None  # 是否佩戴口罩


data_dir = '/home/mext/Desktop/detect_and_recognize_mask_face/datas/DL/mask'
input_img_dir = '/home/mext/Desktop/detect_and_recognize_mask_face/1.png'
# 实例化识别的类

face_recognition = face2.Recognition()
img = cv2.imread(input_img_dir)
face = face_recognition.identify(img)
folder_list = os.listdir(data_dir)
distance = []
num = 0
for i in range(len(folder_list)):
    face_list_class = Face
    face_list = []
    sub_folder = os.path.join(data_dir, folder_list[i])
    sub_folder_list = os.listdir(sub_folder)
    if len(sub_folder_list) >= 5:
        for j in range(5):  # 每个人只计算了
            img1 = cv2.imread(os.path.join(sub_folder, sub_folder_list[j]))
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            face_list_class.image = img1
            emb = face_recognition.encoder.generate_embedding(face_list_class)
            # face1 = face_recognition.identify(img1)
            distance0 = np.linalg.norm(face[0].embedding - emb)

            num += 1
            print(str(distance0) + '***' + str(num))
            distance.append(distance0)

plt.plot(distance)
plt.show()


