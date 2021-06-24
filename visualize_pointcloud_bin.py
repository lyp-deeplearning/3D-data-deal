## 点云数据显示bin文件
import os
import numpy as np
import struct
import open3d
import mayavi.mlab
import torch


def read_bin_velodyne(path):
    pc_list=[]
    with open(path,'rb') as f:
        content=f.read()
        pc_iter=struct.iter_unpack('ffff',content)
        for idx,point in enumerate(pc_iter):
            pc_list.append([point[0],point[1],point[2]])
    return np.asarray(pc_list,dtype=np.float32)


def viz_mayavi(points,vals="distance"):
    x=points[:,0]
    y=points[:,1]
    z=points[:,2]
    r=points[:,3]
    d=torch.sqrt(x**2+y**2)

    if vals=="height":
        col=z
    else:
        col=d

    fig=mayavi.mlab.figure(bgcolor=(0,0,0),size=(1280,720))
    mayavi.mlab.points3d(x,y,z,
                         col,
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )

    mayavi.mlab.show()



def main():
    pc_dir="D:\CODE_UBT\LX\mmdetection3d\demo\data\sunrgbd\\sunrgbd_000017.bin"
    # pcd=open3d.open3d.geometry.PointCloud()
    # example_bin=read_bin_velodyne(pc_dir)
    #
    # # from numpy to open3d
    # pcd.points=open3d.open3d.utility.Vector3dVector(example_bin)
    # open3d.open3d.visualization.draw_geometries([pcd])

    ### mayavi
    mypointcloud = np.fromfile(pc_dir, dtype=np.float32, count=-1).reshape([-1, 4])
    mypointcloud = torch.from_numpy(mypointcloud)
    print(mypointcloud.size())
    print(mypointcloud.type())
    viz_mayavi(mypointcloud, vals="height")


if __name__=="__main__":
    main()
