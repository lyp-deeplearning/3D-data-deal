import os
import cv2
import numpy as np
# 存放标签的类
class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()
        # content就是一个字符串，根据空格分割开来
        lines = content.split()
        # 去掉空字符
        lines = list(filter(lambda x: len(x), lines))
        self.name, self.truncated, self.occluded, self.alpha = lines[0], float(lines[1]), float(lines[2]), float(
            lines[3])
        self.bbox = [lines[4], lines[5], lines[6], lines[7]]
        self.bbox = np.array([float(x) for x in self.bbox])
        self.dimensions = [lines[8], lines[9], lines[10]]
        self.dimensions = np.array([float(x) for x in self.dimensions])
        # for kitti
        self.location = [lines[11], lines[12], lines[13]]
        self.location = np.array([float(x) for x in self.location])
        self.rotation_y = float(lines[14])











# 存放矫正参数的类
class Calib:
    def __init__(self, dict_calib):
        super(Calib, self).__init__()
        #self.P0 = dict_calib['P0'].reshape((3, 4))
        #self.P1 = dict_calib['P1'].reshape((3, 4))
        self.P2 = dict_calib['P2'].reshape((3, 4))
        self.P2=self.P2[:3,:3]
        #self.P2=self.P2.T
       # self.P3 = dict_calib['P3'].reshape((3, 4))
        #self.R0_rect = dict_calib['R0_rect'].reshape((3, 3))
       # self.Tr_velo_to_cam = dict_calib['Tr_velo_to_cam'].reshape((3, 4))
        #self.Tr_imu_to_velo = dict_calib['Tr_imu_to_velo'].reshape((3, 4))

def bboxes2d_in_rgb(img, objects):
    img_2d, img_3d = img, img
    for obj in objects:
        # 为每个物体都要绘制2d矩形框
        cv2.rectangle(img_2d, (int(obj.bbox[0]), int(obj.bbox[1])), (int(obj.bbox[2]), int(obj.bbox[3])),
                      color=(0, 255, 0),
                      thickness=2)
    # 显示图像
    cv2.imshow("2d_bboxes", img_2d)
    cv2.waitKey(0)


# 根据偏航角计算旋转矩阵（逆时针旋转）
def rot_y(rotation_y):
    cos = np.cos(rotation_y)
    sin = np.sin(rotation_y)
    R = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return R

def rot_z(rotation_y):
    c=np.cos(rotation_y)
    s=np.sin(rotation_y)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R






def bboxes_in_rgb(img,objects,calib):
    img_2d = img
    img_3d = img.copy()
    for obj in objects:


        cv2.rectangle(img_2d, (int(obj.bbox[0]), int(obj.bbox[1])), (int(obj.bbox[2]), int(obj.bbox[3])),
                      color=(0, 255, 0),
                      thickness=2)

        #######################################################################
        # 2 get the roll matrix
        # 得到旋转矩阵
        R = rot_y(obj.rotation_y)
        h, w, l = obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]

        x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y = [0, 0, 0, 0, -h, -h, -h, -h]
        # y = [-h / 2, -h / 2, -h / 2, -h / 2, h / 2, h / 2, h / 2, h / 2]
        z = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]







        # 得到目标物体经过旋转之后的实际尺寸（得到其在相机坐标系下的实际尺寸）
        corner_3d = np.vstack([x, y, z])
        corner_3d = np.dot(R, corner_3d)

        # 将该物体移动到相机坐标系下的原点处（涉及到坐标的移动，直接相加就行）
        corner_3d[0, :] += obj.location[0]
        corner_3d[1, :] += obj.location[1]
        corner_3d[2, :] += obj.location[2]

        # 将3d的bbox转换到2d坐标系中（需要用到内参矩阵)
        #corner_3d = np.vstack((corner_3d, np.zeros((1, corner_3d.shape[-1]))))
        corner_2d = np.dot(calib.P2, corner_3d)
        # 在像素坐标系下，横坐标x = corner_2d[0, :] /= corner_2d[2, :]
        # 纵坐标的值以此类推
        corner_2d[0, :] /= corner_2d[2, :]
        corner_2d[1, :] /= corner_2d[2, :]

        # 将浮点数据都转成整型数据
        corner_2d = np.array(corner_2d, dtype=np.int)
        # 绘制立方体边界框
        for corner_i in range(0, 4):
            i, j = corner_i, (corner_i + 1) % 4
            cv2.line(img_3d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]),
                     color=(0, 255, 0),
                     thickness=2)
            i, j = corner_i + 4, (corner_i + 1) % 4 + 4
            cv2.line(img_3d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]),
                     color=(0, 255, 0),
                     thickness=2)
            i, j = corner_i, corner_i + 4
            cv2.line(img_3d, (corner_2d[0, i], corner_2d[1, i]), (corner_2d[0, j], corner_2d[1, j]),
                     color=(0, 255, 0),
                     thickness=2)

    cv2.imshow("2d_bboxes", img_3d)
    cv2.waitKey(0)
    #cv2.imwrite("D:\CODE_UBT\\f-point结果\sun-rgbd结果\\1.jpg",img_3d)









if __name__ == '__main__':
    # 用来可视化数据集中数据
    # 输入数据集的地址
    dir_path = './data/KITTI'
    split = "training"
    ##### deal kitti style data ####

    # calib_path = "C:/Users/12088/Desktop/kitty_demo/CAL_007479.txt"
    # # RGB图像的文件夹地址
    # images_path = "C:/Users/12088/Desktop/kitty_demo/007413.png"
    # # 标签文件夹的地址
    # labels_path = "C:/Users/12088/Desktop/kitty_demo/007413.txt"
    calib_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\calib_new3\\002068.txt"  #calib_new10
    # RGB图像的文件夹地址
    images_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\image\\002068.jpg"
    # 标签文件夹的地址
    labels_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\label_new3\\002068.txt"  #label_new10





    with open(labels_path) as f:
        lines = f.readlines()
    # 去掉空行和换行
    lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))
    obj_3d_label= [Object3d(x) for x in lines]


    # 读取文件，每一个都表示一个参数矩阵，我们用Calib这个类来表示
    with open(calib_path) as f:
        lines = f.readlines()
    # 去掉空行和换行符
    lines = list(filter(lambda x: len(x) and x != '\n', lines))
    dict_calib = {}
    for line in lines:
        key, value = line.split(':')
        dict_calib[key] = np.array([float(x) for x in value.split()])
    calib=Calib(dict_calib)

    img=cv2.imread(images_path)
    #bboxes2d_in_rgb(img,objects=obj_3d_label)
    bboxes_in_rgb(img,objects=obj_3d_label,calib=calib)





