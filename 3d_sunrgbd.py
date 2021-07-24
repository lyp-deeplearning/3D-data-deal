import os
import cv2
import numpy as np
# 存放标签的类
class Object3d:
    def __init__(self, content):
        super(Object3d, self).__init__()


        ################
        data=content.split(' ')
        data[1:]=[float(x) for x in data[1:]]
        self.xmin = data[1]
        self.ymin = data[2]
        self.xmax = data[1] + data[3]
        self.ymax = data[2] + data[4]
        self.bbox = np.array([self.xmin, self.ymin, self.xmax, self.ymax])
        self.location = np.array([data[5], data[6], data[7]])
        #self.unused_dimension = np.array([data[8], data[9], data[10]])
        self.w = data[8]
        self.l = data[9]
        self.h = data[10]


        self.orientation = np.zeros((3,))
        self.orientation[0] = data[11]
        self.orientation[1] = data[12]
        self.rotation_y = -1 * np.arctan2(self.orientation[1], self.orientation[0])






# 存放矫正参数的类
class Calib:
    def __init__(self, calib_path):
        super(Calib, self).__init__()
        lines = [line.rstrip() for line in open(calib_path)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3, 3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3, 3), order='F')
        self.f_u = self.K[0, 0]
        self.f_v = self.K[1, 1]
        self.c_u = self.K[0, 2]
        self.c_v = self.K[1, 2]


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

        l=obj.l
        w=obj.w
        h=obj.h
        x = [-l,l,l,-l,-l,l,l,-l]
        y = [w,w,-w,-w,w,w,-w,-w]
        z = [h,h,h,h,-h,-h,-h,-h]
        corner_3d = np.vstack([x, y, z])

        ##1
        R = rot_z(-1 * obj.rotation_y)
        corners_3d = np.dot(R, corner_3d)

        #2
        corners_3d[0, :] += obj.location[0]
        corners_3d[1, :] += obj.location[1]
        corners_3d[2, :] += obj.location[2]
        #3
        corner_3d_1 = np.dot(np.transpose(calib.Rtilt), corners_3d)
        corner_3d_1=np.transpose(corner_3d_1)
        corner_3d_2 = np.copy(corner_3d_1)
        corner_3d_2[:, [0, 1, 2]] = corner_3d_2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
        corner_3d_2[:, 1] *= -1
        #4
        corner_2d = np.dot(corner_3d_2, np.transpose(calib.K))
        corner_2d[:, 0] /= corner_2d[:, 2]
        corner_2d[:, 1] /= corner_2d[:, 2]



        # 将浮点数据都转成整型数据
        corner_2d = np.array(corner_2d, dtype=np.int)
        # 绘制立方体边界框
        for corner_i in range(0, 4):
            i, j = corner_i, (corner_i + 1) % 4
            cv2.line(img_3d, (corner_2d[i, 0], corner_2d[i, 1]), (corner_2d[j, 0], corner_2d[j, 1]),
                     color=(0, 255, 0),
                     thickness=2)
            i, j = corner_i + 4, (corner_i + 1) % 4 + 4
            cv2.line(img_3d, (corner_2d[i, 0], corner_2d[i, 1]), (corner_2d[j, 0], corner_2d[j, 1]),
                     color=(0, 255, 0),
                     thickness=2)
            i, j = corner_i, corner_i + 4
            cv2.line(img_3d, (corner_2d[i, 0], corner_2d[i, 1]), (corner_2d[j, 0], corner_2d[j, 1]),
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

    calib_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\calib\\002068.txt"  #calib_new10
    # RGB图像的文件夹地址
    images_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\image\\002068.jpg"
    # 标签文件夹的地址
    labels_path = "D:\CODE_UBT\datasets\sun-rgbd\sunrgbd_trainval\label\\002068.txt"  #label_new10





    with open(labels_path) as f:
        lines = f.readlines()
    # 去掉空行和换行
    lines = list(filter(lambda x: len(x) > 0 and x != '\n', lines))
    obj_3d_label= [Object3d(x) for x in lines]

    calib=Calib(calib_path)



    img=cv2.imread(images_path)

    bboxes_in_rgb(img,objects=obj_3d_label,calib=calib)





