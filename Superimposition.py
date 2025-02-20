###################################################
###              ICP and RANSAC                 ###
###################################################
# Author: Anika Kofod Petersen
# Date: 1st June 2024
# Extraordinary Dependencies: open3d
# Tested using python 3.10.6


# import libraries
import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import stl
import argparse


def parse_stl(stl_file):
    """ get info from stl file (memory effecient)"""
    # Load the STL file
    mesh_data = stl.Mesh.from_file(stl_file)

    # Extract vertices and faces
    vertices = mesh_data.vectors.reshape((-1, 3))
    faces = np.arange(len(vertices)).reshape((-1, 3))

    return vertices, faces

#SOURCE CODE: https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
def preprocess_point_cloud(pcd, voxel_size=0.1, fpfh=False):
    """ Downsample and calculate fpfh"""
    # Downsample
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # Estimate normals
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Calculate FPFH
    pcd_fpfh = None
    if fpfh == True:
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh
    
def ICP(path1,path2, threshold = 5):
    """ ICP point-to-plane registration """
    # Data Load
    source_vertices, source_faces = parse_stl(path2)
    target_vertices, target_faces = parse_stl(path1)
    trans_init = np.asarray([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0], [0, 0, 0, 0]])
    trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                [-0.139, 0.967, -0.215, 0.7],
                [0.487, 0.255, 0.835, -1.4],
                [0.0, 0.0, 0.0, 1.0]])

    #Create point cloud
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_vertices)
    target.points = o3d.utility.Vector3dVector(target_vertices)

    # Downsample
    source, __ = preprocess_point_cloud(source)
    target, __ = preprocess_point_cloud(target)

    #Point-to-plane ICP
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg_p2l.transformation, reg_p2l.fitness, reg_p2l.inlier_rmse

def RANSAC(path1,path2, distance_threshold = 5):
    """ RANSAC registration """
    # Data Load
    source_mesh = o3d.io.read_triangle_mesh(path2)
    target_mesh = o3d.io.read_triangle_mesh(path1)

    #Create point cloud
    source = o3d.geometry.PointCloud()
    target = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_vertices)
    target.points = o3d.utility.Vector3dVector(target_vertices)
    source.points = source_mesh.vertices
    target.points = target_mesh.vertices

    # Downsample
    source, source_fpfh = preprocess_point_cloud(source, fpfh=True)
    target, target_fpfh = preprocess_point_cloud(target, fpfh=True)

    # RANSAC registration
    reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=222, confidence=0.95)) #max_validation=0.999
    return reg_ransac.transformation, reg_ransac.fitness, reg_ransac.inlier_rmse

def label_checker(path1,path2):
    id1 = path1.split("/")[-1].split("_")[0]
    id2 = path2.split("/")[-1].split("_")[0]
    jaw1 = path1.split("/")[-1].split("_")[-2].split("_")[0]
    jaw2 = path2.split("/")[-1].split("_")[-2].split("_")[0]
    cut1 = path1.split("/")[-1].split(".")[0][-2:]
    cut2 = path2.split("/")[-1].split(".")[0][-2:]
    
    if jaw1 == jaw2:
        if id1 == id2:
            if cut1 == cut2 or cut1 == "f0" or cut2 == "f0":
                if path1 == path2:
                    return 2
                else:
                    return 1
            elif cut1 == "p1" and cut2 in ["p1","p3","p4"]:
                if path1 == path2:
                    return 2
                else:
                    return 1
            elif cut1 == "p2" and cut2 in ["p2","p3","p4"]:
                if path1 == path2:
                    return 2
                else:
                    return 1
            elif cut1 == "p3" and cut2 in ["p1","p2","p3"]:
                if path1 == path2:
                    return 2
                else:
                    return 1
            elif cut1 == "p4" and cut2 in ["p1","p2","p4"]:
                if path1 == path2:
                    return 2
                else:
                    return 1
            else:
                return 0
        else:
            return 0
    else:
        return None


# The actual script for running
if __name__ == '__main__':
    
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',
                    default='./path/',
                    help="path to stl file",
                    )
    parser.add_argument('-o', '--out',
                    default='./outpath/',
                    help="path to output file file",
                    )
    args = parser.parse_args()
    
    # Get file paths
    checkpoint_files = ["partial_fitness_match.txt","partial_inliner_rmse_match.txt","partial_fitness_semimatch.txt","partial_inliner_rmse_semimatch.txt","partial_fitness_mismatch.txt","partial_inliner_rmse_mismatch.txt"]
    out = "/path/to/Results/ICP/"
    FPA = "/path/to/Data/A/"
    FPB = "/path/to/Data/B/"
    fpa = [FPA+f for f in os.listdir(FPA)]
    fpb = [FPB+f for f in os.listdir(FPB)]
    fpc = fpa+fpb
    
    fitness_match = []
    inliner_rmse_match = []
    fitness_mismatch = []
    inliner_rmse_mismatch = []
    fitness_semimatch = []
    inliner_rmse_semimatch = []
    
    i = 0
    file1 = args.path
    
    for j, file2 in enumerate(fpc):
        #print(f"{j+1}    {i+1}/{len(fpc)}  {round(((i+1)/len(fpc))*100,2)} %   ", end="\r")
        label = label_checker(file1,file2)
        T, fit, rmse = ICP(file1,file2)
        if label != None:
            if label == 1:
                fitness_semimatch.append(fit)
                inliner_rmse_semimatch.append(rmse)
            elif label == 2:
                fitness_match.append(fit)
                inliner_rmse_match.append(rmse)
            elif label == 0:
                fitness_mismatch.append(fit)
                inliner_rmse_mismatch.append(rmse)
    with open(out+checkpoint_files[0], "a") as f:
        f.write(f'{fitness_match}\n')
    with open(out+checkpoint_files[1], "a") as f:
        f.write(f'{inliner_rmse_match}\n')
    with open(out+checkpoint_files[2], "a") as f:
        f.write(f'{fitness_semimatch}\n')
    with open(out+checkpoint_files[3], "a") as f:
        f.write(f'{inliner_rmse_semimatch}\n')
    with open(out+checkpoint_files[4], "a") as f:
        f.write(f'{fitness_mismatch}\n')
    with open(out+checkpoint_files[5], "a") as f:
        f.write(f'{inliner_rmse_mismatch}\n')
    with open(out+"files_done.txt", "a") as f:
        f.write(f'{file1}\n')

    
    
    

