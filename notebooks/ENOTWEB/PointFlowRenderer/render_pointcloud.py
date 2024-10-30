import numpy as np
import open3d as o3d
import mitsuba as mi

mi.set_variant('scalar_rgb')
import matplotlib.pyplot as plt
from mitsuba import ScalarTransform4f as T
# 功能函数定义
def standardize_bbox(pcl, points_per_object, arg_scale=1.0):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.0
    scale = np.amax(maxs - mins) / arg_scale
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result

def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.linalg.norm(vec)
    vec /= norm
    return vec.tolist()

rectangle_with_bsdf = {
    'type': 'rectangle',
    'bsdf': {
        'type': 'ref',
        'id': 'surfaceMaterial'
    },
    'to_world': T.translate([0, 0, -0.5])@T.scale([10, 10, 1])
    # mi.Transform4f.compose([
    #     mi.Transform4f.scale([10, 10, 1]),
    #     mi.Transform4f.translate([0, 0, -0.5])
    # ])
}

rectangle_with_emitter = {
    'type': 'rectangle',
    'to_world': 
        T.look_at(
            origin=[-4, 4, 20],
            target=[0, 0, 0],
            up=[0, 0, 1]
        ) @ T.scale([10, 10, 1]),
    'emitter': {
        'type': 'area',
        'radiance': {
            'type': 'rgb',
            'value': [6, 6, 6]
        }
    }
}

# 主程序
def run(ply_file_path):
    # 加载 PLY 点云
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    xyz = np.asarray(point_cloud.points)
    xyz = standardize_bbox(xyz, 2048, arg_scale=1.0)  # 标准化点云
    pcd_color = np.asarray(point_cloud.colors)
    # 构建场景字典
    scene_dict = {
        'type': 'scene',
        'integrator': {
            'type': 'path',
            'max_depth': -1
        },
        'sensor': {
            'type': 'perspective',
            'far_clip': 100,
            'near_clip': 0.1,
            'to_world': mi.Transform4f.look_at(
                origin=[3, 3, 3],
                target=[0, 0, 0],
                up=[0, 0, 1]
            ),
            'fov': 25,
            'sampler': {
                'type': 'ldsampler',
                'sample_count': 256
            },
            'film': {
                'type': 'hdrfilm',
                'width': 1600,
                'height': 1200,
                'rfilter': {'type': 'gaussian'},
                'banner': False
            }
        },
        'bsdf': {
            'type': 'roughplastic',
            'id': 'surfaceMaterial',
            'distribution': 'ggx',
            'alpha': 0.05,
            'int_ior': 1.46,
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [1, 1, 1]
            }
        },
        # 'emitter': {
        #     'type': 'area',
        #     'radiance': {
        #         'type': 'rgb',
        #         'value': [6, 6, 6]
        #     }
        # }  # Assuming you have an HDR environment map
    }

    # 添加点云中的每个点作为一个小球体到场景字典中
    for i in range(xyz.shape[0]):
        # point_color = colormap(pcd_color[i, 0], pcd_color[i, 1], pcd_color[i, 2])
        diffuse_bsdf_dict = {
            'type': 'diffuse',
            'bsdf': {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'rgb',
                    'value': pcd_color[i, :].tolist()
                }
            }
        }
        scene_dict.update({'shapes_{}'.format(i):{
            'type': 'sphere',
            'radius': 0.025,
            'center': xyz[i].tolist(),
            'bsdf': diffuse_bsdf_dict['bsdf']
        }})
      
    scene_dict.update({'shape_rect_bsdf': rectangle_with_bsdf})
    scene_dict.update({'shape_rect_emitter': rectangle_with_emitter})
    # 加载场景并渲染
    scene = mi.load_dict(scene_dict)
    image = mi.render(scene, spp=256)
    mi.util.write_bitmap('./results/images/test.png', image)
    print("save image /result/images/test.png")

    # # 显示渲染结果
    # plt.figure(figsize=(10, 8))
    # plt.imshow(mi.core.Bitmap(image).convert_to_rgb())
    # plt.axis('off')
    # plt.show()

# 运行程序
if __name__ == '__main__':
    ply_file_path = './vr_pcd/mug_feat2.ply'  # 替换为你的 PLY 文件路径
    run(ply_file_path)
