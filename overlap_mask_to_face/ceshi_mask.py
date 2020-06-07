import wear_mask
from PIL import Image
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from lib.core.api.facer import FaceAna


# image = cv2.imread('test_img/01.jpg')
image = cv2.imread('test_img/Aishwarya_Rai_0001.png')
facer = FaceAna()
boxes, landmarks, _ = facer.run(image)
face_image = wear_mask.mask_img('test_img/Aishwarya_Rai_0001.png', 'images', landmarks)
# cv2.imshow(face_image)
# face_image = np.array(face_image)
face_image.show()

