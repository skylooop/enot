import numpy as np
import sys, os, subprocess
import OpenEXR
import Imath
from PIL import Image
import imageio

PATH_TO_MITSUBA2 = "/home/m_bobrin/anaconda3/envs/jax/lib/python3.10/site-packages/mitsuba/mitsuba"  # mitsuba exectuable

# replaced by command line arguments
# PATH_TO_NPY = 'pcl_ex.npy' # the tensor to load

# note that sampler is changed to 'independent' and the ldrfilm is changed to hdrfilm
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
            <integer name="width" value="1500"/>
            <integer name="height" value="1024"/>
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


def colormap(x, y, z):
    vec = np.array([x, y, z])
    vec = np.clip(vec, 0.001, 1.0)
    norm = np.sqrt(np.sum(vec ** 2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def standardize_bbox(pcl, points_per_object):
    pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
    np.random.shuffle(pt_indices)
    pcl = pcl[pt_indices]  # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = (mins + maxs) / 2.
    scale = np.amax(maxs - mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center) / scale).astype(np.float32)  # [-0.5, 0.5]
    return result


# only for debugging reasons
def writeply(vertices, ply_file):
    sv = np.shape(vertices)
    points = []
    for v in range(sv[0]):
        vertex = vertices[v]
        points.append("%f %f %f\n" % (vertex[0], vertex[1], vertex[2]))
    print(np.shape(points))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    end_header
    %s
    ''' % (len(vertices), "".join(points)))
    file.close()


# as done in https://gist.github.com/drakeguan/6303065
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
    # rgb8 = [Image.fromarray(c.astype(int)) for c in rgb]
    Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)


def rotation(points, theta):
    R = np.asarray([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    new_points = points @ R
    return new_points


def main(argv):
    if (len(argv) < 2):
        print('filename to npy/ply is not passed as argument. terminated.')
        return

    pathToFile = argv[1]

    degrees = argv[2]

    for indx, deg in enumerate(range(0, int(degrees), 6)):
        radian = deg * np.pi / 180
        filename, file_extension = os.path.splitext(pathToFile)
        filename = filename.split('/')[-1]

        folder = "/home/m_bobrin/ENOTWEB/rotation_car" #os.path.dirname(pathToFile)
        #filename = os.path.basename(pathToFile)

        # for the moment supports npy and ply
        if (file_extension == '.npy'):
            pclTime = np.load(pathToFile)
            pclTimeSize = np.shape(pclTime)
        elif (file_extension == '.npz'):
            pclTime = np.load(pathToFile)
            pclTime = pclTime['pred']
            pclTimeSize = np.shape(pclTime)
        elif (file_extension == '.ply'):
            ply = PlyData.read(pathToFile)
            vertex = ply['vertex']
            (x, y, z) = (vertex[t] for t in ('x', 'y', 'z'))
            pclTime = np.column_stack((x, y, z))
        else:
            print('unsupported file format.')
            return

        if (len(np.shape(pclTime)) < 3):
            pclTimeSize = [1, np.shape(pclTime)[0], np.shape(pclTime)[1]]
            pclTime.resize(pclTimeSize)

        for pcli in range(0, pclTimeSize[0]):
            pcl = pclTime[pcli, :, :]

            pcl_old = standardize_bbox(pcl, pcl.shape[0])
            pcl_old = pcl_old[:, [2, 0, 1]]
            pcl = rotation(pcl_old, radian)
            # pcl = pcl[:, [2, 0, 1]]
            # pcl[:, 1] *= -1
            # pcl[:, 2] += 0.0125

            xml_segments = [xml_head]
            for i in range(pcl.shape[0]):
                color = colormap(pcl_old[i, 0] + 0.5, pcl_old[i, 1] + 0.5, pcl_old[i, 2] + 0.5 - 0.0125)
                xml_segments.append(xml_ball_segment.format(pcl[i, 0], pcl[i, 1], pcl[i, 2], *color))
            xml_segments.append(xml_tail)

            xml_content = str.join('', xml_segments)

            xmlFile = ("%s/%s_%03d.xml" % (folder, filename, indx))
            print(xmlFile)
            with open(xmlFile, 'w') as f:
                f.write(xml_content)
            f.close()

            exrFile = ("%s/%s_%03d.exr" % (folder, filename, indx))
            print(exrFile)
            if (not os.path.exists(exrFile)):
                print(['Running Mitsuba, writing to: ', xmlFile])
                subprocess.run([PATH_TO_MITSUBA2, xmlFile])
            else:
                print('skipping rendering because the EXR file already exists')
                # print(['Running Mitsuba, writing to: ', xmlFile])
                # subprocess.run([PATH_TO_MITSUBA2, xmlFile])

            png = ("%s/images/%s_%03d.png" % (folder, filename, indx))
            print(['Converting EXR to PNG...'])
            pointcloud_image = imageio.imread(exrFile)
            imageio.imwrite(png, pointcloud_image, quality=100)

            #ConvertEXRToJPG(exrFile, png)


if __name__ == "__main__":
    main(sys.argv)