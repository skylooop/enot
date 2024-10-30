"""
    example:
    conda activate gaussian_splatting
    python render_pointcloud_anim.py ./vr_pcd/mug_feat1.ply ./vr_pcd/mug_feat_corr.ply 140 
    生成mug_feat1.ply到mug_feat_corr.ply转换的140帧动画

"""
import numpy as np
import open3d as o3d
import sys
def linear_interpolation(P1, P2, t):
    """
    线性插值函数。
    
    参数:
    - P1: 起始点云，形状为 (n, 3)。
    - P2: 终点点云，形状为 (n, 3)。
    - t: 插值参数，范围为 [0, 1]，其中 0 表示完全是 P1，1 表示完全是 P2。
    
    返回:
    - 插值后的点云。
    """
    return P1 + t * (P2 - P1)

def create_animation(pcd1, pcd2, m, save_dir):
    """
    创建动画并保存每一帧为 PLY 文件。
    
    参数:
    - P1: 起始点云的 numpy 数组，形状为 (n, 3)。
    - P2: 终点点云的 numpy 数组，形状为 (n, 3)。
    - m: 帧数。
    - save_dir: 保存 PLY 文件的目录。
    """
    P1 = np.asarray(pcd1.points)
    P2 = np.asarray(pcd2.points)
    c1 = np.asarray(pcd1.colors)
    frame_names = []
    for i in range(m):
        t = i / (m - 1)  # 计算插值参数 t
        P_interpolated = linear_interpolation(P1, P2, t)  # 获取插值点云
        point_cloud = o3d.geometry.PointCloud()  # 创建 Open3D 点云对象
        point_cloud.points = o3d.utility.Vector3dVector(P_interpolated)  # 设置点云数据
        point_cloud.colors = o3d.utility.Vector3dVector(c1)
        ply_save_path = f"{save_dir}/frame_{i:03d}.ply"
        o3d.io.write_point_cloud(ply_save_path, point_cloud)  # 保存为 PLY 文件
        frame_names.append(ply_save_path)
    return frame_names

if __name__ == "__main__":
    ply1 = sys.argv[1]
    ply2 = sys.argv[2]

    # 1.生成中间帧的点云, 保存ply并且返回ply文件名的list(arr)
    frame_num = int(sys.argv[3])  # 动画帧数
    # 假设 P1 和 P2 是你的两个 (n, 3) 点云 numpy 数组
    P1 = o3d.io.read_point_cloud(ply1)
    P2 = o3d.io.read_point_cloud(ply2)
    save_dir = '/home/m_bobrin/ENOTWEB/PointFlowRenderer/Anim'  # 保存 PLY 文件的目录
    arr = create_animation(P1, P2, frame_num, save_dir)
    # 2.渲染为image
    from render_pointcloud_sphere import run
    for ply in arr:
        run(ply)  


