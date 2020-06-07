from PIL import Image
import numpy as np
import random
import cv2
from overlap_mask_to_face.lib.core.api.facer import FaceAna
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# /home/mext/datasets/lfw/lfw_mtcnnpy_160
input_dir = '/home/mext/Desktop/detect_and_recognize_mask_face/datas/DL/cropped'
output_dir = '/home/mext/Desktop/detect_and_recognize_mask_face/datas/DL/mask'
mask_img_dir = '/home/mext/Desktop/detect_and_recognize_mask_face/overlap_mask_to_face/images'


def mask_img(ori_img_dir, mask_img_dir, face_landmarks):
    """
    :param ori_img_dir: 包含人脸的原图路径
    :param mask_img_dir: 口罩图片路径
    :param face_landmarks: 检测出的人脸关键点
        face_landmarks为人脸的68个稠密关键点
        face_landmarks.shape = [1,68,2]
    :return:
    """

    # 载入人脸图片和口罩图片
    face_pic = Image.open(ori_img_dir)
    mask_img_name_list = os.listdir(mask_img_dir)
    # 随机选择一张口罩图片
    mask_pic = Image.open(
        os.path.join(mask_img_dir, mask_img_name_list[random.randint(0, len(mask_img_name_list) - 1)]))

    # 获取鼻子和脸颊的特征点
    nose_point = face_landmarks[0, 29, :]
    nose_vector = np.array(nose_point)
    chin_left_point = face_landmarks[0, 2, :]
    chin_right_point = face_landmarks[0, 14, :]
    chin_bottom_point = face_landmarks[0, 8, :]
    chin_bottom_vector = np.array(chin_bottom_point)

    # 拆分、缩放和合并口罩
    width = mask_pic.width
    height = mask_pic.height
    width_ration = 1.4  # 宽度的缩放系数
    # 获取鼻梁上的点到脸底点的距离
    new_height = int(np.linalg.norm(nose_vector - chin_bottom_vector))

    # 调整左口罩大小，宽度为脸左点到中心线的距离 * 宽度系数
    mask_left_img = mask_pic.crop((0, 0, width // 2, height))
    # 获取脸左点到中心线的距离
    mask_left_width = get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
    mask_left_width = int(mask_left_width * width_ration)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # 调整右口罩大小，宽度为脸右点到中心线的距离 * 宽度系数
    mask_right_img = mask_pic.crop((width // 2, 0, width, height))
    # 获取脸右点到中心线的距离
    mask_right_width = get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
    mask_right_width = int(mask_right_width * width_ration)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # 左右合并为新口罩
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_pic_emp = Image.new('RGBA', size)
    mask_pic_emp.paste(mask_left_img, (0, 0), mask_left_img)
    mask_pic_emp.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # 旋转口罩
    angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
    rotated_mask_pic_emp = mask_pic_emp.rotate(angle, expand=True)

    # 将口罩图片放到合适的位置
    center_x = (nose_point[0] + chin_bottom_point[0]) // 2
    center_y = (nose_point[1] + chin_bottom_point[1]) // 2

    offset = mask_pic_emp.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_pic_emp.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_pic_emp.height // 2

    # 添加口罩
    face_pic.paste(mask_pic_emp, (int(box_x), int(box_y)), mask_pic_emp)

    return face_pic


def get_distance_from_point_to_line(point, line_point1, line_point2):
    distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                      (line_point1[0] - line_point2[0]) * point[1] +
                      (line_point2[0] - line_point1[0]) * line_point1[1] +
                      (line_point1[1] - line_point2[1]) * line_point1[0]) / \
               np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                       (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
    return int(distance)


folder_name_list = os.listdir(input_dir)
len_folder_name_list = len(folder_name_list)
num_count = 0
facer = FaceAna()

for i in range(len_folder_name_list):

    sub_folder_path = os.path.join(input_dir, folder_name_list[i])
    image_name_list = os.listdir(sub_folder_path)
    # len_image_name_list = len(image_name_list)

    for image_name in image_name_list:

        # image = Image.open(os.path.join(sub_folder_path, image_name))
        image = cv2.imread(os.path.join(sub_folder_path, image_name))
        # facer = FaceAna()
        _, landmarks, _ = facer.run(image)

        if len(landmarks.shape) == 3:
            face_image = mask_img(os.path.join(sub_folder_path, image_name), mask_img_dir, landmarks)

            save_path_name = os.path.join(output_dir, os.path.join(folder_name_list[i], image_name))
            save_path = os.path.join(output_dir, folder_name_list[i])
            if not os.path.exists(save_path):
                os.mkdir(save_path)
                face_image.save(save_path_name)
            else:
                print("目录已存在，直接保存了")
                face_image.save(save_path_name)
            num_count += 1
            print("处理完成第" + str(num_count) + "张人脸，加油！")
        else:
            print(os.path.join(sub_folder_path, image_name) + "有问题，去检查")
            continue


