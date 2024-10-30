from hashlib import new
import open3d as o3d
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
import os
import sys

import open3d as o3d
import numpy as np
import imageio
import OpenEXR
import Imath
from PIL import Image
import subprocess

def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale) # [-0.5, 0.5]
    return result

def configure_xml(radius=0.025):
    xml_head = \
    """
<scene version="0.6.0">
    <integrator type="path">
        <integer name="maxDepth" value="-1"/>
    </integrator>
    <sensor type="perspective">
        <float name="farClip" value="100"/>
        <float name="nearClip" value="0.1"/>
        <transform name="toWorld">
            <lookat origin="3,3,3" target="0,0,0" up="0,0,1"/>
        </transform>
        <float name="fov" value="25"/>
        
        <sampler type="ldsampler">
            <integer name="sampleCount" value="256"/>
        </sampler>
        <film type="hdrfilm">
            <integer name="width" value="1600"/> 
            <integer name="height" value="1200"/>
            <rfilter type="gaussian"/>
            <boolean name="banner" value="false"/>
        </film>
    </sensor>
    
    <bsdf type="roughplastic" id="surfaceMaterial">
        <string name="distribution" value="ggx"/>
        <float name="alpha" value="0.05"/>
        <float name="intIOR" value="1.46"/>
        <rgb name="diffuseReflectance" value="1,1,1"/> <!-- default 0.5 -->
    </bsdf>
    
"""
    # 1600x1200
    # 0.025
    xml_ball_segment = \
    """
        <shape type="sphere">
            <float name="radius" value="0.025"/>
            <transform name="toWorld">
                <translate x="{}" y="{}" z="{}"/>
            </transform>
            <bsdf type="diffuse">
                <rgb name="reflectance" value="{},{},{}"/>
            </bsdf>
        </shape>
    """

    xml_tail = \
    """
    <shape type="rectangle">
        <ref name="bsdf" id="surfaceMaterial"/>
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <translate x="0" y="0" z="-0.5"/>
        </transform>
    </shape>
    
    <shape type="rectangle">
        <transform name="toWorld">
            <scale x="10" y="10" z="1"/>
            <lookat origin="-4,4,20" target="0,0,0" up="0,0,1"/>
        </transform>
        <emitter type="area">
            <rgb name="radiance" value="6,6,6"/>
        </emitter>
    </shape>
</scene>
"""
    return xml_head, xml_ball_segment, xml_tail

def colormap(x,y,z):
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]

def ConvertEXRToJPG(exrfile, jpgfile):
    File = OpenEXR.InputFile(exrfile)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

    rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in 'RGB']
    for i in range(3):
        rgb[i] = np.where(rgb[i] <= 0.0031308,
                          (rgb[i] * 12.92) * 255.0,
                          (1.055 * (rgb[i] ** (1.0 / 2.4)) - 0.055) * 255.0)

    rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "PNG", quality=100)

# generate_xml: 是否检查已经生成xml文件
def render_image(ply_file_path, scene_path, image_path, use_existed_xml=False):
    filename = ply_file_path.split('/')[-1].split('.')[0]
    xml_file_path = os.path.join(scene_path, "{}.xml".format(filename))
    point_cloud = o3d.io.read_point_cloud(ply_file_path)
    # R = rotate_point_cloud(point_cloud, -115, center=[0, 0, 0], axis='x')
    # point_cloud.rotate(R, center=(0, 0, 0))
    # R = rotate_point_cloud(point_cloud, 90, center=[0, 0, 0], axis='z')
    # point_cloud.rotate(R, center=(0, 0, 0))

    if os.path.exists(xml_file_path) and use_existed_xml:
        print("xml file {} already exists!".format(xml_file_path))
    else:
        print("xml file {} does not exist, creating...".format(xml_file_path))  
        # Configure xml
        xml_head, xml_ball_segment, xml_tail = configure_xml()
        # Load a PLY point cloud
        xyz = np.asarray(point_cloud.points)
        xyz = standardize_bbox(xyz, xyz.shape[0])
        xyz = xyz[:,[2,0,1]]
        xyz[:,0] *= -1
        xyz[:,2] += 0.0125
    
        xml_segments = [xml_head]
        for i in range(xyz.shape[0]):
            color = colormap(xyz[i,0]+0.5,xyz[i,1]+0.5,xyz[i,2]+0.5-0.0125)
            xml_segments.append(xml_ball_segment.format(xyz[i,0],xyz[i,1],xyz[i,2], *color))
        xml_segments.append(xml_tail)
        
        xml_content = str.join('', xml_segments)
        
        with open(xml_file_path, 'w') as f:
            f.write(xml_content)
        f.close()

    # 读取xml文件
    subprocess.run(["/home/m_bobrin/anaconda3/envs/jax/lib/python3.10/site-packages/mitsuba/mitsuba", xml_file_path])
    # mi.set_variant("scalar_rgb")
    # scene = mi.load_file(xml_file_path)

    # print("load xml file from {}".format(xml_file_path))
    # image = mi.render(scene)
    # # Convert the image to a NumPy array
    # image_np = np.array(image)

    # # Apply contrast adjustment
    # minimum = np.min(image_np)
    # maximum = np.max(image_np)
    # contrast_adjusted_image = np.round((image_np - minimum) / (maximum - minimum) * 255 * 1.3) # Adjust the multiplier for contrast

    # # Ensure values are within valid range
    # contrast_adjusted_image = np.clip(contrast_adjusted_image, 0, 255)

    #ConvertEXRToJPG(xml_file_path.split(".")[0] + '.exr', f"/home/m_bobrin/ENOTWEB/PointFlowRenderer/Anim/images/{filename}.png")
    
    # Save the image
    # image_save_to = os.path.join(image_path, "{}.png".format(filename))
    # mi.util.write_bitmap(image_save_to, contrast_adjusted_image.astype(np.uint8))
    # bmp = mi.Bitmap(image, pixel_format=mi.Bitmap.PixelFormat.RGB)
    # image = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
    # image_save_to = os.path.join(image_path, "{}.png".format(filename))
    # mi.util.write_bitmap(image_save_to, image)

    pointcloud_image = imageio.imread(xml_file_path.split(".")[0] + '.exr')
    imageio.imwrite(f"/home/m_bobrin/ENOTWEB/PointFlowRenderer/Anim/images/{filename}.png", pointcloud_image, quality=100)

def run(ply_file_path):
    scene_path = '/home/m_bobrin/ENOTWEB/PointFlowRenderer/Anim/results/'
    image_path = '/home/m_bobrin/ENOTWEB/PointFlowRenderer/Anim/images/'
    render_image(ply_file_path, scene_path, image_path, use_existed_xml=False)

if __name__ == '__main__':
    run(sys.argv[1])
