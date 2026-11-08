import json
import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation as R


def load_data(file_path):
    """加载标定数据，包括内参、外参、图像路径等"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    matrices = []           # robot base -> end-effector
    homogeneous_matrices = []  # camera -> tag
    image_paths = []
    
    for entry in data['data']: 
        matrices.append(np.array(entry['matrix']))
        homogeneous_matrices.append(np.array(entry['homogeneous_matrix']))
        image_paths.append(entry['image'])
    
    # 加载内参
    intrinsics = data['intrinsics']
    camera_matrix = np.array([
        [intrinsics['fx'], 0, intrinsics['ppx']],
        [0, intrinsics['fy'], intrinsics['ppy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 畸变系数（通常 Realsense rectified 图像为 0）
    dist_coeffs = np.array(intrinsics.get('coeffs', [0, 0, 0, 0, 0]), dtype=np.float64)
    if len(dist_coeffs) < 5:
        dist_coeffs = np.pad(dist_coeffs, (0, 5 - len(dist_coeffs)))
    
    return matrices, homogeneous_matrices, image_paths, camera_matrix, dist_coeffs


def rotate_camera_around_world_axis(cam2base, euler_angles_deg, position_fixed=True):
    """
    让相机绕世界坐标系的X、Y、Z轴旋转指定角度
    
    Args:
        cam2base: 4x4 变换矩阵 [R|t]
        euler_angles_deg: [rx, ry, rz] 分别表示绕X、Y、Z轴的旋转角度（度）
        position_fixed: 是否保持位置不变，只改变朝向
    
    Returns:
        旋转后的4x4变换矩阵
    """
    # 创建旋转对象
    rotation = R.from_euler('XYZ', euler_angles_deg, degrees=True)
    R_world = rotation.as_matrix()
    
    # 分离旋转和平移
    R_original = cam2base[:3, :3]
    t_original = cam2base[:3, 3]
    
    if position_fixed:
        # 保持位置不变，只改变朝向
        R_new = R_world @ R_original
        t_new = t_original
    else:
        # 同时旋转位置和朝向
        R_new = R_world @ R_original
        t_new = R_world @ t_original
    
    # 构建新的变换矩阵
    cam2base_new = np.eye(4)
    cam2base_new[:3, :3] = R_new
    cam2base_new[:3, 3] = t_new
    
    return cam2base_new


def get_camera_ray_intersection_with_plane(cam2base, plane_z=0.0):
    """
    计算相机光轴与指定Z平面的交点
    
    Args:
        cam2base: 4x4 相机到世界坐标系的变换矩阵
        plane_z: 平面的Z坐标 (默认为0，即桌面)
    
    Returns:
        交点坐标 [x, y, z]，如果无交点返回 None
    """
    # 相机在世界坐标系中的位置
    camera_pos = cam2base[:3, 3]
    
    # 相机光轴方向（相机Z轴在世界坐标系中的方向）
    camera_z_axis = cam2base[:3, 2]  # 第三列是Z轴方向
    
    print(f"相机位置: {camera_pos}")
    print(f"光轴方向: {camera_z_axis}")
    print(f"光轴Z分量: {camera_z_axis[2]:.6f}")
    
    # 检查是否平行于平面
    if abs(camera_z_axis[2]) < 1e-8:
        print("光轴平行于Z=0平面，无交点")
        return None
    
    # 计算交点参数 t
    # 相机位置 + t * 光轴方向 = 交点
    # camera_pos.z + t * camera_z_axis.z = plane_z
    t = (plane_z - camera_pos[2]) / camera_z_axis[2]
    
    # 如果 t < 0，说明交点在相机后方（背向方向）
    if t < 0:
        print("交点在相机后方（光轴背向平面）")
        # 仍然返回交点，但可以标记为无效
    
    # 计算交点坐标
    intersection_point = camera_pos + t * camera_z_axis
    intersection_point[2] = plane_z  # 确保Z坐标精确为plane_z
    
    print(f"交点参数 t: {t:.6f}")
    print(f"交点坐标: {intersection_point}")
    
    return intersection_point

def fix_tag_pose(tag_pose_cam):
    """
    修正 AprilTag 坐标系，使其 Z 轴与相机光轴方向一致（朝下看）
    绕 X 轴旋转 180°：[x, y, z] -> [x, -y, -z]
    """
    # 旋转矩阵：绕 X 轴旋转 180°
    Rx_180 = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float64)
    
    R = tag_pose_cam[:3, :3]
    t = tag_pose_cam[:3, 3]
    
    # 更新旋转矩阵
    R_corrected = R @ Rx_180  # 顺序：先应用原始旋转，再修正
    # 平移向量不变（因为旋转中心在原点）
    
    corrected_pose = np.eye(4)
    corrected_pose[:3, :3] = R_corrected
    corrected_pose[:3, 3] = t
    return corrected_pose

def perform_hand_eye_calibration(matrices, homogeneous_matrices):
    """执行手眼标定（eye-to-hand）"""
    # Robot: gripper (end-effector) to base
    R_gripper2base = [np.array(matrix[:3,:3]) for matrix in matrices]
    t_gripper2base = [np.array(matrix[:3, 3]) for matrix in matrices]
    
    # 转换为 base to gripper（OpenCV 要求）
    R_base2gripper = [R.T for R in R_gripper2base]
    t_base2gripper = [-R.T @ t for R, t in zip(R_gripper2base, t_gripper2base)]

    # Camera: target (tag) to camera
    R_target2cam = [np.array(matrix[:3,:3]) for matrix in homogeneous_matrices]
    t_target2cam = [np.array(matrix[:3, 3]) for matrix in homogeneous_matrices]
    
    # 手眼标定：求解 camera to base
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper, 
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI  # 可选其他方法
    )
    
    cam2base = np.eye(4)
    cam2base[:3, :3] = R_cam2base
    cam2base[:3, 3] = t_cam2base.flatten()
    
    return cam2base

def compute_reprojection_error(image_path, tag_pose_cam, cam2base, base2gripper, camera_matrix, tag_size=0.055):
    """
    计算重投影误差（验证标定质量）
    
    注意：这里我们用 AprilTag 的 3D 角点 + camera pose 重投影到图像，
    但也可以用 base -> tag 的完整链路验证
    """
    # AprilTag 的 3D 角点（在 tag 坐标系中）
    s = tag_size / 2.0
    tag_3d_points = np.array([
        [-s, -s, 0],
        [ s, -s, 0],
        [ s,  s, 0],
        [-s,  s, 0]
    ], dtype=np.float64)
    
    # 从 base 到 tag 的完整变换
    gripper2base = base2gripper  # 4x4
    cam2gripper = np.linalg.inv(cam2base)  # 假设相机固定在 base（eye-to-hand）
    # 实际上：base -> cam -> tag
    # 所以 base -> tag = cam2base @ (camera -> tag)
    base2tag = cam2base @ tag_pose_cam
    
    # 将 3D 点变换到相机坐标系
    tag_points_cam = (tag_pose_cam[:3, :3] @ tag_3d_points.T + tag_pose_cam[:3, 3:4]).T
    
    # 投影到图像
    projected_points, _ = cv2.projectPoints(
        tag_points_cam,
        np.zeros(3), np.zeros(3),  # 无额外旋转平移（已在 tag_points_cam 中）
        camera_matrix,
        np.zeros(5)  # 假设无畸变（rectified image）
    )
    
    projected_points = projected_points.reshape(-1, 2)
    
    # 从图像中检测实际角点（用于计算误差）
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 这里简化：直接使用 AprilTag 检测的角点（你原始数据中没存，所以跳过）
    # 实际应用中，建议在采集时保存 corners
    
    # 如果无法获取真实角点，可跳过误差计算
    return None

def main():
    # name = "front"
    # name = "left"
    name = "right"
    file_path = f'./data_{name}.json'
    if not os.path.exists(file_path):
        print(f"错误: {file_path} 不存在")
        return
    
    matrices, homogeneous_matrices, image_paths, camera_matrix, dist_coeffs = load_data(file_path)
    
    print("✅ 相机内参:")
    print(camera_matrix)
    print("畸变系数:", dist_coeffs)
    
    print(f"\n📊 加载 {len(matrices)} 组标定样本")
    
    if len(matrices) < 3:
        print("警告: 标定样本少于3组，结果可能不可靠")


    # homogeneous_matrices = [fix_tag_pose(p) for p in homogeneous_matrices]
    
    # 执行手眼标定
    cam2base = perform_hand_eye_calibration(matrices, homogeneous_matrices)
    
    get_camera_ray_intersection_with_plane(cam2base, 0.02)
    # cam2base = rotate_camera_around_world_axis(cam2base, [0, 180, 0])
    
    print("\n✅ 手眼标定结果 (Camera to Base):")
    print(cam2base)
    
    # 保存结果
    np.save(f'cam2base_{name}.npy', cam2base)
    print("\n💾 结果已保存为 'cam2base.npy'")
    
    # 可选：打印逆变换（Base to Camera）
    base2cam = np.linalg.inv(cam2base)
    print("\n🔄 Base to Camera:")
    print(base2cam)

if __name__ == "__main__":
    main()