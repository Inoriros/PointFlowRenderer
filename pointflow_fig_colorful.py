import torch
import mitsuba as mi
import pypose as pp
import numpy as np
import open3d as o3d
import copy
from point_cloud_viewer import read_xyz, read_off, read_pcd, read_bin_pc


# --------------------------------- xml file settings -------------------------------------
# <lookat origin="3, 3, 3" target="0,0,0" up="0,0,1"/>

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
        <lookat origin="2, 2, 2" 
                target="-0.0003292992369946558,
                        0.00039989585003839931,
                        -0.0011882884369665044" 
                up="0.25403083996730591, -0.30849108431533662, 0.91667965137416074"/>
        </transform>
        <float name="fov" value="35"/>
        
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

xml_ball_segment = \
"""
    <shape type="sphere">
        <float name="radius" value="0.0030"/>
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
# --------------------------------- xml file settings ------------------------------------


# ------------------------ Function to generate Mitsuba scene file ------------------------
def standardize_bbox(pcl, points_per_object):
    if pcl.shape[0] < points_per_object:
        print("Point cloud should have at least {} points. All the points will be used this time.".format(points_per_object))
        points_per_object = pcl.shape[0]
    if points_per_object > 0:
        pt_indices = np.random.choice(pcl.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt_indices)
        pcl = pcl[pt_indices] # n by 3
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result = ((pcl - center)/scale).astype(np.float32) # [-0.5, 0.5]
    # result = ((pcl)/scale).astype(np.float32) # [-0.5, 0.5]

    return result

def standardize_bbox_twoPCs(pcl1, pcl2, points_per_object):
    if np.min([pcl1.shape[0], pcl2.shape[0]]) < points_per_object:
        print("Point clouds should have at least {} points. All the points of the small point cloud will be used this time.".format(points_per_object))
        points_per_object = np.min([pcl1.shape[0], pcl2.shape[0]])
    if points_per_object > 0:
        pt1_indices = np.random.choice(pcl1.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt1_indices)
        pcl1 = pcl1[pt1_indices] # n by 3
        pt2_indices = np.random.choice(pcl2.shape[0], points_per_object, replace=False)
        np.random.shuffle(pt2_indices)
        pcl2 = pcl2[pt2_indices] # n by 3

    pcl = np.concatenate((pcl1, pcl2), axis=0)
    mins = np.amin(pcl, axis=0)
    maxs = np.amax(pcl, axis=0)
    center = ( mins + maxs ) / 2.
    scale = np.amax(maxs-mins)
    print("Center: {}, Scale: {}".format(center, scale))
    result1 = ((pcl1 - center)/scale).astype(np.float32) # [-0.5, 0.5]
    result2 = ((pcl2 - center)/scale).astype(np.float32) # [-0.5, 0.5]

    return result1, result2

def generate_transformation_matrix(rot_angles, translation_vec):
    """
    Generate a transformation matrix given rotation angles and a translation vector.

    Parameters:
    rot_angles (np.array): Rotation angles along each axis in radians. Shape (3,).
    translation_vec (np.array): Translation vector. Shape (3,).

    Returns:
    np.array: 4x4 transformation matrix.
    """
    # Create rotation matrix from euler angles
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(rot_angles[0]), -np.sin(rot_angles[0])],
        [0, np.sin(rot_angles[0]), np.cos(rot_angles[0])]
    ])

    rot_y = np.array([
        [np.cos(rot_angles[1]), 0, np.sin(rot_angles[1])],
        [0, 1, 0],
        [-np.sin(rot_angles[1]), 0, np.cos(rot_angles[1])]
    ])

    rot_z = np.array([
        [np.cos(rot_angles[2]), -np.sin(rot_angles[2]), 0],
        [np.sin(rot_angles[2]), np.cos(rot_angles[2]), 0],
        [0, 0, 1]
    ])

    rot_matrix = rot_z @ rot_y @ rot_x

    # Create 4x4 transformation matrix
    trans_matrix = np.eye(4)
    trans_matrix[:3, :3] = rot_matrix
    trans_matrix[:3, 3] = translation_vec

    return trans_matrix

def transform_pointcloud(src_points_ori, transform):
    '''
    args:
        src_points_ori: [N, 3]
        transform: [4, 4]
    return: 
        result_points: [N, 3]
    '''
    transform = torch.tensor(transform)
    src_points_ori = torch.tensor(src_points_ori)
    src_points_homo = torch.cat([src_points_ori, torch.ones(src_points_ori.shape[0],1)], dim=1)
    result_points = torch.matmul(transform, src_points_homo.T).T
    result_points = result_points[..., :, :3]
    return np.array(result_points)

def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    combined = source_temp + target_temp

    o3d.io.write_point_cloud("combined_point_cloud.pcd", combined)

    # Visualize the point cloud and capture the image
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    vis.add_geometry(combined)

    vis.run()  # Adjust the view as you want before taking the screenshot

    # vis.capture_screen_image("point_cloud_view2.png")

    vis.destroy_window()

def colormap(x,y,z):
    '''
    args: 
    '''
    vec = np.array([x,y,z])
    vec = np.clip(vec, 0.001,1.0)
    norm = np.sqrt(np.sum(vec**2))
    vec /= norm
    return [vec[0], vec[1], vec[2]]


def iICP_vis_rendering(model_type='3DLoMatch', scene_name='Home_1_0_31', addition_transf=0, o3d_view=0, rot=[0, 60*np.pi/180, 60*np.pi/180], reg_method='re', save_image=None):
    xml_segments = [xml_head]
    if reg_method == 're' or reg_method == 'refined':
        method = 'refined'
    else:
        method = 'estimated'
    # Replace 'file.txt' with your actual file path
    if reg_method == 'gt':
        transformation_matrix = np.loadtxt('/home/jared/Desktop/iICP_pc_transformation_results/' + model_type + '_' + scene_name + '.npz_after_ICP_gt_transform.txt')
    else:
        transformation_matrix = np.loadtxt('/home/jared/Desktop/iICP_pc_transformation_results/' + model_type + '_' + scene_name + '.npz_' + method + '_transform.txt')


    # Transform pointcloud for better visualization
    if addition_transf == 1:
        rot = np.array(rot)
        trans = np.array([0, 0, 0])
        vis_transformation_matrix = generate_transformation_matrix(rot, trans)
        # rot = np.array([-40*np.pi/180, 0, 0])
        # # rot = np.array([-40*np.pi/180, -10*np.pi/180, -10*np.pi/180])
        # trans = np.array([0, 0, 0])
        # vis_transformation_matrix1 = generate_transformation_matrix(rot, trans)
        # vis_transformation_matrix = vis_transformation_matrix1 @ vis_transformation_matrix


    
    # pointcloud 1 - reference point cloud
    pcd1 = o3d.io.read_point_cloud('/home/jared/Desktop/iICP_pc_transformation_results/' + model_type + '_' + scene_name + '.npz_ref_points.ply')
    pcl1 = np.asarray(pcd1.points)
    if addition_transf == 1:
        pcl1 = transform_pointcloud(pcl1, vis_transformation_matrix)

    # pointcloud 2 - source point cloud
    pcd2 = o3d.io.read_point_cloud('/home/jared/Desktop/iICP_pc_transformation_results/' + model_type + '_' + scene_name + '.npz_src_points.ply')
    pcl2 = np.asarray(pcd2.points)
    if reg_method == 're' or reg_method == 'es' or reg_method == 'refined' or reg_method == 'estimated' or reg_method == 'gt':
        pcl2 = transform_pointcloud(pcl2, transformation_matrix)
    if addition_transf == 1:
        pcl2 = transform_pointcloud(pcl2, vis_transformation_matrix)

    # Standardize point clouds
    pcl1, pcl2 =standardize_bbox_twoPCs(pcl1, pcl2, -1)
    pcl1 = pcl1[:,[2,0,1]]
    pcl1 *= -1
    pcl2 = pcl2[:,[2,0,1]]
    pcl2 *= -1

    # O3d visualization
    if o3d_view == 1:
        pcd01 = o3d.geometry.PointCloud()
        pcd01.points = o3d.utility.Vector3dVector(pcl1)
        pcd02 = o3d.geometry.PointCloud()
        pcd02.points = o3d.utility.Vector3dVector(pcl2)
        draw_registration_result(pcd01, pcd02)
    

    # poincloud 1 
    for i in range(pcl1.shape[0]):
        color = (0.172, 0.498, 0.722)  # Blue #99d8c9
        xml_segments.append(xml_ball_segment.format(pcl1[i,0],pcl1[i,1],pcl1[i,2], *color))

    # pointcloud 2
    for i in range(pcl2.shape[0]):
        # color2 = (0.788, 0.58, 0.78)  # light purple #c994c7
        color2 = (0.996, 0.698, 0.298)  # Orange #c994c7
        xml_segments.append(xml_ball_segment.format(pcl2[i,0],pcl2[i,1],pcl2[i,2], *color2))

    xml_segments.append(xml_tail)
    xml_content = str.join('', xml_segments)
    with open('mitsuba_scene.xml', 'w') as f:
        f.write(xml_content)
    with open('./iICP_vis_xml/' + model_type + '_' + scene_name + '_' + reg_method + '_whole.xml', 'w') as f:
        f.write(xml_content)
    
    print('The Mitsuba rendering scene file has been saved as: ' + './iICP_vis_xml/' + model_type + '_' + scene_name + '_' + reg_method + '_whole.xml')\
    
    if save_image != None:
        mi.variants()
        mi.set_variant("scalar_rgb")        
        scene = mi.load_file('./iICP_vis_xml/' + model_type + '_' + scene_name + '_' + reg_method + '_whole.xml')
        img_ref = mi.render(scene, spp=64)

        bmp_small = mi.util.convert_to_bitmap(img_ref)
        bmp_small.write('./iICP_vis_results/' + model_type + '_' + scene_name + '_' + reg_method + '_whole.' + save_image)          

# ------------------------ Function to generate Mitsuba scene file ------------------------


# ----------------------- New Point cloud registration --------------------------------

if __name__ == '__main__':
    # model_type = 'ModelNet'
    # scene_name = '1759'
    model_type = '3DMatch'
    scene_name = 'Study_51_53'
    iICP_vis_rendering(model_type, scene_name, rot=[60*np.pi/180, 0, 60*np.pi/180]
                       , addition_transf=0, o3d_view=0, reg_method='estimated', save_image='jpeg')

# 


# # ----------------------- All Point clouds registration --------------------------------
#     model_types = ['3DLoMatch', '3DMatch']
#     scene_names_Lo = ['Home_1_0_31', 'Home_1_9_24', 'Kitchen_11_29']
#     scene_names = ['Hotel_2_13_27', 'Hotel_2_13_29', 'Kitchen_0_1', 'Kitchen_1_31', 'MIT_Lab_0_11']

    # # generate Mitsuba scene files
    # count = 0
    # for model_type in model_types:
    #     if model_type == '3DLoMatch':
    #         for scene_name in scene_names_Lo:
    #             iICP_vis_rendering(model_type, scene_name, addition_transf=0, o3d_view=0)
    #             count += 1
    #             # print the count number
    #             print('The ' + str(count) + 'th Mitsuba rendering scene file has been saved.')

    #     elif model_type == '3DMatch':
    #         for scene_name in scene_names:
    #             iICP_vis_rendering(model_type, scene_name, addition_transf=0, o3d_view=0)
    #             count += 1
    #             # print the count number
    #             print('The ' + str(count) + 'th Mitsuba rendering scene file has been saved.')
    #     else:
    #         print('Wrong model type!')
    #         break
    
    # # generate images using Mitsuba renderer and save them
    # mi.variants()
    # mi.set_variant("scalar_rgb")

    # resolution = 64
    # count = 0
    # for model_type in model_types:
    #     if model_type == '3DLoMatch':
    #         for scene_name in scene_names_Lo:
    #             scene = mi.load_file('./iICP_vis_xml/' + model_type + '_' + scene_name + '_whole.xml')
    #             img_ref = mi.render(scene, spp=resolution)
    #             bmp_small = mi.util.convert_to_bitmap(img_ref)
    #             bmp_small.write('./iICP_vis_results_lowRes/' + model_type + '_' + scene_name + '_whole.jpeg')
    #             count += 1
    #             # print the count number
    #             print('The ' + str(count) + 'th Mitsuba rendering scene file has been saved.')

    #     elif model_type == '3DMatch':
    #         for scene_name in scene_names:
    #             scene = mi.load_file('./iICP_vis_xml/' + model_type + '_' + scene_name + '_whole.xml')
    #             img_ref = mi.render(scene, spp=resolution)
    #             bmp_small = mi.util.convert_to_bitmap(img_ref)
    #             bmp_small.write('./iICP_vis_results_lowRes/' + model_type + '_' + scene_name + '_whole.jpeg')
    #             count += 1
    #             # print the count number
    #             print('The ' + str(count) + 'th Mitsuba rendering scene file has been saved.')
    #     else:
    #         print('Wrong model type!')
    #         break



# # ------------------------------- Point cloud registration --------------------------------
# xml_segments = [xml_head]
# # pcl = np.load('chair_pcl.npy')
# # pcl, faces = read_off('/home/jared/SAIR_Lab/Super-Map/data/PU-GAN/simple/hand.off')

# # pointcloud 1
# pcd = o3d.io.read_point_cloud('source_point_cloud.pcd')
# pcl = np.asarray(pcd.points)

# pcl = standardize_bbox(pcl, 8000)
# pcl = pcl[:,[2,0,1]]
# pcl[:,0] *= -1
# pcl[:,2] += 0.0125

# # pointcloud 2
# pcd2 = o3d.io.read_point_cloud('target_point_cloud.pcd')
# pcl2 = np.asarray(pcd2.points)

# pcl2 = standardize_bbox(pcl2, 8000)
# pcl2 = pcl2[:,[2,0,1]]
# pcl2[:,0] *= -1
# pcl2[:,2] += 0.0125

# # # original color map - purple to yellow
# # for i in range(pcl.shape[0]):
# #     color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
# #     xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
# # xml_segments.append(xml_tail)


# # poincloud 1 
# for i in range(pcl.shape[0]):
#     color = (0.172, 0.498, 0.722)  # Blue #99d8c9
#     xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))

# # pointcloud 2
# for i in range(pcl2.shape[0]):
#     # color2 = (0.788, 0.58, 0.78)  # light purple #c994c7
#     color2 = (0.996, 0.698, 0.298)  # Orange #c994c7
#     xml_segments.append(xml_ball_segment.format(pcl2[i,0],pcl2[i,1],pcl2[i,2], *color2))


# xml_segments.append(xml_tail)

# xml_content = str.join('', xml_segments)

# with open('mitsuba_scene.xml', 'w') as f:
#     f.write(xml_content)


# # --------------------- single point cloud ---------------------
# xml_segments = [xml_head]

# # pcl = np.load('chair_pcl.npy')
# pcl, faces = read_off('/home/jared/SAIR_Lab/Super-Map/data/PU-GAN/simple/hand.off')

# # # pointcloud 1
# # pcd = o3d.io.read_point_cloud('source_point_cloud.pcd')
# # pcl = np.asarray(pcd.points)

# # Transform pointcloud
# rot = np.array([180*np.pi/180, 180*np.pi/180, 180*np.pi/180])
# trans = np.array([0, 0, 0])
# t_mat = generate_transformation_matrix(rot, trans)


# pcl = transform_pointcloud(pcl, t_mat)

# pcl = standardize_bbox(pcl, 66000)
# pcl = pcl[:,[2,0,1]]
# pcl[:,0] *= -1
# pcl[:,2] += 0.0125

# # original color map - purple to yellow
# for i in range(pcl.shape[0]):
#     color = colormap(pcl[i,0]+0.5,pcl[i,1]+0.5,pcl[i,2]+0.5-0.0125)
#     xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
# xml_segments.append(xml_tail)

# # # poincloud 1 
# # for i in range(pcl.shape[0]):
# #     color = (0.172, 0.498, 0.722)  # Blue #99d8c9
# #     xml_segments.append(xml_ball_segment.format(pcl[i,0],pcl[i,1],pcl[i,2], *color))
# # xml_segments.append(xml_tail)

# xml_content = str.join('', xml_segments)

# with open('mitsuba_scene.xml', 'w') as f:
#     f.write(xml_content)