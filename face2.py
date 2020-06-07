import pickle
import os

import cv2
import numpy as np
import tensorflow as tf
from scipy import misc
from mtcnn_src.mtcnn import mtcnn
from facenet_src import facenet
from mobilenet_src import mobilenet
from keras.applications.imagenet_utils import preprocess_input
import mtcnn_src.utils as utils

facenet_model_checkpoint = 'models/20200505-085843'
classifier_model = 'models/dl_classifier_masks_65.pkl'


def getitem(list1, level=0):
    for item in list1:
        if isinstance(item, list):
            getitem(item, level + 1)
        else:
            for tab in range(level):
                print('\t', end='')  # 输出一个制表符，并且将 print 后面的换行符去掉，这样就是了缩进
            print(item)


class Face:
    def __init__(self):
        self.name = None
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.ifwear = None  # 是否佩戴口罩


class Recognition:

    def __init__(self):
        self.detect = Detection()
        self.encoder = Encoder()
        self.identifier = Identifier()
        self.wearornot = Wearornot()

    def add_identity(self, image, person_name):
        faces = self.detect.find_faces(image)
        '''
        faces中包含内容（每一张人脸都对应下面的内容）
                人脸图片的名字
                bounding-box的四个坐标
                人脸图像
                原图像
                特征向量
        '''

        if len(faces) == 1:
            face = faces[0]
            face.name = person_name
            face.embedding = self.encoder.generate_embedding(face)
            return faces

    def identify(self, image):
        faces = self.detect.find_faces(image)
        '''
        faces中包含内容（每一张人脸都对应下面的内容）
                人脸图片的名字
                bounding-box的四个坐标
                人脸图像
                原图像
                特征向量
        '''
        for i, face in enumerate(faces):
            face.embedding = self.encoder.generate_embedding(face)
            face.name = self.identifier.identify(face)
            face.ifwear = self.wearornot.if_wear(face.image)

        return faces


class Identifier:
    def __init__(self):
        # 打开分类器模型，读入模型和分类名
        with open(classifier_model, 'rb') as infile:
            self.model, self.class_names = pickle.load(infile)

    def identify(self, face):
        if face.embedding is not None:
            predictions = self.model.predict_proba([face.embedding])
            best_class_indices = np.argmax(predictions, axis=1)
            return self.class_names[best_class_indices[0]]


class Encoder:
    def __init__(self):
        self.sess = tf.Session()
        # 加载facenet模型
        with self.sess.as_default():
            facenet.load_model(facenet_model_checkpoint)

    # 生成特征向量
    def generate_embedding(self, face):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        prewhiten_face = facenet.prewhiten(face.image)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
        return self.sess.run(embeddings, feed_dict=feed_dict)[0]


class Detection:
    def __init__(self):
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5, 0.6, 0.8]
        self.face_crop_margin = 32
        self.face_crop_size = 160

    def find_faces(self, image):
        faces = []
        # height, width, _ = np.shape(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bounding_boxes = self.mtcnn_model.detectFace(image, self.threshold)

        for bb in bounding_boxes:
            # bb的前四个为bounding-box的坐标
            face = Face()
            face.container_image = image  # 这个是原图
            face.bounding_box = np.zeros(4, dtype=np.int32)  # 这个是bounding-box的四个值

            img_size = np.asarray(image.shape)[0:2]  # 原图片尺寸
            # bounding-box的四个坐标,扩宽一点且不超过图片的范围
            face.bounding_box[0] = np.maximum(bb[0] - self.face_crop_margin // 2, 0)
            face.bounding_box[1] = np.maximum(bb[1] - self.face_crop_margin // 2, 0)
            face.bounding_box[2] = np.minimum(bb[2] + self.face_crop_margin // 2, img_size[1])
            face.bounding_box[3] = np.minimum(bb[3] + self.face_crop_margin // 2, img_size[0])
            # 把人脸图片裁出来
            cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]

            # # 上面监测的人脸可能是长方形
            # rectangles_temp = utils.rect2square(np.array(face.bounding_box, dtype=np.int32))
            # rectangles_temp[:, 0] = np.clip(rectangles_temp[:, 0], 0, width)  # 不失真的方式转换为正方形
            # rectangles_temp[:, 1] = np.clip(rectangles_temp[:, 1], 0, height)
            # rectangles_temp[:, 2] = np.clip(rectangles_temp[:, 2], 0, width)
            # rectangles_temp[:, 3] = np.clip(rectangles_temp[:, 3], 0, height)

            #
            #
            # # 调整成需要的尺寸
            face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')
            # face.image = image[rectangles_temp[1]:rectangles_temp[3], rectangles_temp[0]:rectangles_temp[2], :]
            faces.append(face)
            '''
            faces中包含内容(有*号的为当前已有内容)
                    人脸图片的名字
                    bounding-box的四个坐标**
                    人脸图像**
                    原图像**
                    特征向量
            '''
        return faces


class Wearornot:
    def __init__(self):
        self.NUM_CLASSES = 2
        self.classes_name = ['mask', 'nomask']
        self.mask_model = mobilenet.MobileNet(input_shape=[160, 160, 3], classes=self.NUM_CLASSES)
        self.mask_model.load_weights('models/logslast_one_65.h5')

    def if_wear(self, new_img):
        new_img = preprocess_input(
            np.reshape(np.array(new_img, np.float64), [1, 160, 160, 3]))
        classes = self.classes_name[int(np.argmax(self.mask_model.predict(new_img)[0]))]
        return classes
