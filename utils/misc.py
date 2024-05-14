import torch
import numpy as np
import random
import os
import open3d as o3d

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def estimate_plane(xyz, normalize=True):
    """
    :param xyz:  3*3 array
    x1 y1 z1
    x2 y2 z2
    x3 y3 z3
    :return: a b c d
      model_coefficients.resize (4);
      model_coefficients[0] = p1p0[1] * p2p0[2] - p1p0[2] * p2p0[1];
      model_coefficients[1] = p1p0[2] * p2p0[0] - p1p0[0] * p2p0[2];
      model_coefficients[2] = p1p0[0] * p2p0[1] - p1p0[1] * p2p0[0];
      model_coefficients[3] = 0;
      // Normalize
      model_coefficients.normalize ();
      // ... + d = 0
      model_coefficients[3] = -1 * (model_coefficients.template head<4>().dot (p0.matrix ()));
    """
    vector1 = xyz[1,:] - xyz[0,:]
    vector2 = xyz[2,:] - xyz[0,:]

    if not np.all(vector1):
        # print('will divide by zero..')
        return None
    dy1dy2 = vector2 / vector1

    if  not ((dy1dy2[0] != dy1dy2[1])  or  (dy1dy2[2] != dy1dy2[1])):
        return None

    a = (vector1[1]*vector2[2]) - (vector1[2]*vector2[1])
    b = (vector1[2]*vector2[0]) - (vector1[0]*vector2[2])
    c = (vector1[0]*vector2[1]) - (vector1[1]*vector2[0])
    # normalize
    if normalize:
        r = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        a = a / r
        b = b / r
        c = c / r
    d = -(a*xyz[0,0] + b*xyz[0,1] + c*xyz[0,2])
    # return a,b,c,d
    return np.array([a,b,c,d])


def my_ransac(data,
              distance_threshold=0.3,
              P=0.99,
              sample_size=3,
              max_iterations=1000,
              ):
    """
    :param data:
    :param sample_size:
    :param P :
    :param distance_threshold:
    :param max_iterations:
    :return:
    """
    max_point_num = -999
    i = 0
    K = 10
    L_data = len(data)
    R_L = range(L_data)

    while i < K:
        s3 = random.sample(R_L, sample_size)

        if abs(data[s3[0],1] - data[s3[1],1]) < 3:
            continue

        coeffs = estimate_plane(data[s3,:], normalize=False)
        if coeffs is None:
            continue

        r = np.sqrt(coeffs[0]**2 + coeffs[1]**2 + coeffs[2]**2 )
        d = np.divide(np.abs(np.matmul(coeffs[:3], data.T) + coeffs[3]) , r)
        d_filt = np.array(d < distance_threshold)
        near_point_num = np.sum(d_filt,axis=0)

        if near_point_num > max_point_num:
            max_point_num = near_point_num

            best_model = coeffs
            best_filt = d_filt

            w = near_point_num / L_data

            wn = np.power(w, 3)
            p_no_outliers = 1.0 - wn

            K = (np.log(1-P) / np.log(p_no_outliers))

        i += 1
        if i > max_iterations:
            print(' RANSAC reached the maximum number of trials.')
            break

    return np.argwhere(best_filt).flatten(), best_model


def range_filter(pcd, dist_min=1, dist_max=50, z_limit=[-2.5, 4]):
    dist = np.sqrt(np.sum(pcd[:, :3] ** 2, axis = 1))
    ego_mask = np.asarray(pcd[:,0]>-2) & np.asarray(pcd[:,0]<2) \
               & np.asarray(pcd[:,1]>-1) & np.asarray(pcd[:,1]<1) \
               & np.asarray(pcd[:,2]>-2) & np.asarray(pcd[:,2]<2) 
    mask = np.asarray(dist >= dist_min) & np.asarray(dist <= dist_max) \
           & np.asarray(pcd[:,2]>z_limit[0]) & np.asarray(pcd[:,2]<z_limit[1]) \
           & ~ego_mask
    pcd = pcd[mask]
    return pcd


def point_removal(pc_raw):
    pc_rm = range_filter(pc_raw)

    pcd_rm = o3d.geometry.PointCloud()
    pcd_rm.points = o3d.utility.Vector3dVector(pc_rm[:,:3])
    pcd_rm, ind = pcd_rm.remove_statistical_outlier(64, 3.0)
    pc_rm = np.asarray(pcd_rm.points)
    
    indices, _ = my_ransac(pc_rm[:, :3], distance_threshold=0.15)
    index_total = indices
    for i in range(5):
        indices, _ = my_ransac(pc_rm[:, :3], distance_threshold=0.15)
        index_total = np.unique(np.concatenate((index_total, indices)))
    indices = index_total

    indices = indices[pc_rm[indices, 2] < -1]
    pc_ground = pc_rm[indices].copy()

    pc_rm[indices] = 999 + 1
    pc_rm = pc_rm[pc_rm[:, 2] <= 999]

    pcd_rm = o3d.geometry.PointCloud()
    pcd_rm.points = o3d.utility.Vector3dVector(pc_rm[:,:3])
    pcd_rm, ind = pcd_rm.remove_statistical_outlier(64, 3.0)
    pc_rm = np.asarray(pcd_rm.points)

    return pc_rm, pc_ground