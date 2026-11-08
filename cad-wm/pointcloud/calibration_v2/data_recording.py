import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
from pupil_apriltags import Detector
import panda_py

    # - "250222073548" # front
    # - "348522075731" # left
    # - "152122075567" # right
serials = {
    "front":    "250222073548",
    "left":     "348522075731",
    "right":    "152122075567",
}
# name = "front"
# name = "left"
name = "right"

# Initialize the robot


# Define tag size in meters
# tag_size = 0.055 #0.04646  
tag_size = 0.08

# Initialize Intel RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(serials[name])
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


# Start streaming
profile = pipeline.start(config)

# Get camera intrinsics dynamically
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
fx, fy = intrinsics.fx, intrinsics.fy
cx, cy = intrinsics.ppx, intrinsics.ppy
camera_params = [fx, fy, cx, cy]

print(f"Camera Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# Initialize AprilTag detector
detector = Detector(families="tag36h11")
# tagStandard41h12 tag25h9

# Create a window
cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)

# Initialize variables
data = []
intrinsics = None
image_counter = 0  # Counter for image filenames

def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Convert to grayscale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags with pose estimation
        detections = detector.detect(gray, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)

        # Draw detection results
        for detection in detections:
            # Draw tag border
            for idx in range(len(detection.corners)):
                pt1 = tuple(map(int, detection.corners[idx]))
                pt2 = tuple(map(int, detection.corners[(idx + 1) % 4]))
                cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

            # Draw center
            center = tuple(map(int, detection.center))
            cv2.circle(color_image, center, 5, (0, 0, 255), -1)

            # Display ID
            cv2.putText(color_image, f"ID: {detection.tag_id}",
                        (center[0] - 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Extract pose
            t = detection.pose_t.flatten()  # Translation vector
            R = detection.pose_R  # Rotation matrix

            # Flip to make camera Z axis point backward (OpenGL style: view direction = -Z)
            # C_flip = np.array([[1,  0,  0],
            #                    [0, -1,  0],
            #                    [0,  0, -1]])
            # t = C_flip @ t_orig
            # R = C_flip @ R_orig

            # Display translation in GUI
            pose_text = f"XYZ: ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m"
            cv2.putText(color_image, pose_text, (center[0] - 50, center[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw coordinate axes (X = red, Y = green, Z = blue)
            axis_length = 0.2  # Reduced axis length for better visualization
            axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])  # X, Y, Z
            axes_img_pts = []
            for i in range(3):
                axis_end = t + R @ axes[i]  # Transform axis endpoints
                x, y = int(fx * axis_end[0] / axis_end[2] + cx), int(fy * axis_end[1] / axis_end[2] + cy)
                axes_img_pts.append((x, y))

            cv2.line(color_image, center, axes_img_pts[0], (0, 0, 255), 2)  # X - Red
            cv2.line(color_image, center, axes_img_pts[1], (0, 255, 0), 2)  # Y - Green
            cv2.line(color_image, center, axes_img_pts[2], (255, 0, 0), 2)  # Z - Blue (pointing outward)

            # Print pose to console
            # print(f"Tag ID: {detection.tag_id}")
            # print(f"Translation (m): x={t[0]:.3f}, y={t[1]:.3f}, z={t[2]:.3f}")
            # print(f"Rotation matrix:\n{R}\n")

            # Create homogeneous transformation matrix
            # 相机坐标到AprilTag坐标系的变换
            homogeneous_matrix = np.eye(4)
            homogeneous_matrix[:3, :3] = R
            homogeneous_matrix[:3, 3] = t

        # Show images
        cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == ord('r'):
            # Generate a random 4x4 matrix
            # 机器人基座到机械臂末端的变换
            hostname = '192.168.1.101'
            panda = panda_py.Panda(hostname)

            matrix = panda.get_pose()
            print("pose_matrix: ", matrix)
            # Save the image to a file
            image_filename = f'./images/{image_counter}.png'
            cv2.imwrite(image_filename, color_image)
            # Save the image filename, random matrix, and homogeneous transformation matrix as a pair
            data.append({
                'image': image_filename,
                'matrix': matrix.tolist(),
                'homogeneous_matrix': homogeneous_matrix.tolist()
            })
            # print(f"Image, matrix, and homogeneous matrix recorded: {image_filename}")
            print(t)
            image_counter += 1  # Increment the counter

        elif key == ord('q'):
            # Get intrinsics
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            intrinsics_data = {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                # 'model': intrinsics.model,
                'coeffs': intrinsics.coeffs
            }
            # Save data to JSON file
            save_to_json(f'data_{name}.json', {'data': data, 'intrinsics': intrinsics_data})
            print("Intrinsics saved and exiting.")
            break

finally:
    # Stop streaming
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics_data = {
        'width': intrinsics.width,
        'height': intrinsics.height,
        'ppx': intrinsics.ppx,
        'ppy': intrinsics.ppy,
        'fx': intrinsics.fx,
        'fy': intrinsics.fy,
        #'model': intrinsics.model,
        'coeffs': intrinsics.coeffs
    }
    pipeline.stop()
    # Save data to JSON file
    save_to_json(f'data_{name}.json', {'data': data, 'intrinsics': intrinsics_data})
    print("Intrinsics saved and exiting.")
    cv2.destroyAllWindows()