# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
import cv2
import face2


# 在图像上标注一些人脸信息和帧率信息
def add_overlays(frame, faces):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name + '-' + face.ifwear, (face_bb[0], face_bb[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                            thickness=1, lineType=1)


if __name__ == '__main__':

    face_recognition = face2.Recognition()
    img = cv2.imread('3.jpg')
    img2 = img.copy()
    faces = face_recognition.identify(img2)
    '''
    faces中包含内容（每一张人脸都对应下面的内容）
            人脸图片的名字
            bounding-box的四个坐标
            人脸图像
            原图像
            特征向量
    '''
    add_overlays(img2, faces)
    cv2.imshow('face', img)
    cv2.imshow('mask-face', img2)
    cv2.imwrite('33.jpg', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

