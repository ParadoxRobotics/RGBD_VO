# One-file Realsense RGBD Visual Odometry for robotics applications:
# Author :  MUNCH Quentin 2023

import numpy as np
import cv2
import imutils
import pyrealsense2 as rs
from matplotlib import pyplot as plt

# General input Configuration
H = 480
W = 640
# ORB feature extractor init
ORB = cv2.ORB_create(nfeatures=3000, nlevels=8, scoreType=cv2.ORB_FAST_SCORE)
print("[I] ORB feature detector")
# LKT matcher init
LKT_config = dict(
    winSize=(25, 25),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
print("[I] KLT initialized")
# PNP solver init
pnp_method = cv2.SOLVEPNP_P3P
pnp_confidence = 0.9999
pnp_retroprojection_error = 1
pnp_iteration = 1000
# Init pose
rotation_matrix = np.eye(3)
translation_vector = np.zeros([[0], [0], [0]])

# Init the Realsense pipeline capture
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, 60)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, 60)
# start Realsense and recover scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# align depth camera to the color camera
alignement = rs.align(rs.stream.color)
# recover calibration data
depth_profile = profile.get_stream(rs.stream.depth)
depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
color_profile = profile.get_stream(rs.stream.color)
color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
# construct intrinsic camera matrix for the depth camera
fx_depth = depth_intrinsics.fx
fy_depth = depth_intrinsics.fy
cx_depth = depth_intrinsics.ppx
cy_depth = depth_intrinsics.ppy
depth_intrinsics_matrix = np.array([[fx_depth, 0, cx_depth], [0, fy_depth, cy_depth], [0, 0, 1]])
# construct intrinsic camera matrix for the color camera
fx_color = color_intrinsics.fx
fy_color = color_intrinsics.fy
cx_color = color_intrinsics.ppx
cy_color = color_intrinsics.ppy
color_intrinsics_matrix = np.array([[fx_color, 0, cx_color], [0, fy_color, cy_color], [0, 0, 1]])
print("[I] Realsense camera initialized")

# flush frame for auto-adjustement
print("[I] Running auto-adjustement...")
for i in range(0, 100):
    # Acquire state
    ref_state = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    ref_aligned_state = alignement.process(ref_state)
    ref_frame_aligned = ref_aligned_state.get_color_frame()
    ref_depth_aligned = ref_aligned_state.get_depth_frame()
print("[I] Auto-adjustement done")

print("[I] Initilize reference and local map")
# get rectified / aligned depth and color frame
depth_ref = np.asanyarray(ref_depth_aligned.get_data()) * depth_scale
frame_ref = cv2.cvtColor(np.asanyarray(ref_frame_aligned.get_data()), cv2.COLOR_BGR2GRAY)
# extract keypoints
kpt_ref, des_ref = ORB.detectAndCompute(frame_ref, None)
# convert keypoint as an array of 2D points and create initial 3D local map
kpt_ref_filter = []
keyframe_map = []
for i in range(len(kpt_ref)):
    if depth_ref[int(kpt_ref[i].pt[1]), int(kpt_ref[i].pt[0])] != 0.0:
        # append keypoint and descriptor
        kpt_ref_filter.append([kpt_ref[i].pt[0], kpt_ref[i].pt[1]])
        # compute 3D point in the aligned color camera
        xyz_point = rs.rs2_deproject_pixel_to_point(
            color_intrinsics,
            [int(kpt_ref[i].pt[1]), int(kpt_ref[i].pt[0])],
            depth_ref[int(kpt_ref[i].pt[1]), int(kpt_ref[i].pt[0])],
        )
        keyframe_map.append(xyz_point)
# convert it into arrays
kpt_ref = np.expand_dims(np.array(kpt_ref_filter), axis=1).astype("float32")  # weird reshaping (n,1,2) dtype=float32
keyframe_map = np.array(keyframe_map)

# main loop
print("[I] Local map initialized. Starting stereo odometry pipeline !")
while True:
    # Acquire state
    cur_state = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    cur_aligned_state = alignement.process(cur_state)
    cur_frame_aligned = cur_aligned_state.get_color_frame()
    cur_depth_aligned = cur_aligned_state.get_depth_frame()
    # get rectified / aligned depth and color frame
    depth_cur = np.asanyarray(cur_depth_aligned.get_data()) * depth_scale
    frame_cur = cv2.cvtColor(np.asanyarray(cur_frame_aligned.get_data()), cv2.COLOR_BGR2GRAY)

    # compute keypoints displacement using KLT tracker
    kpt_cur_pred, status, error = cv2.calcOpticalFlowPyrLK(frame_ref, frame_cur, kpt_ref, None, **LKT_config)
    kpt_ref_pred, status, error = cv2.calcOpticalFlowPyrLK(frame_cur, frame_ref, kpt_cur_pred, None, **LKT_config)
    # compute distance point2point
    dist = abs(kpt_ref - kpt_ref_pred).reshape(-1, 2).max(-1)
    matched_kpt = dist < 1
    # remove unmatched keypoints
    matched_kpt_ref = []
    matched_kpt_cur = []
    matched_keyframe_map = []
    for i, matched in enumerate(matched_kpt):
        if matched:
            matched_kpt_ref.append(kpt_ref[i])
            matched_kpt_cur.append(kpt_cur_pred[i])
            matched_keyframe_map(keyframe_map[i])
    # convert it into arrays
    matched_kpt_ref = np.array(matched_kpt_ref)
    matched_kpt_cur = np.array(matched_kpt_cur)
    # update keyframe map
    keyframe_map = np.array(matched_keyframe_map)

    # compute 3D pose from estimated current keypoint and world coordinate keyframe local map
    inliers, rotation_vector, translation_vector, idxPose = cv2.solvePnPRansac(
        keyframe_map,
        matched_kpt_cur,
        color_intrinsics_matrix,
        None,
        iterationsCount=pnp_iteration,
        reprojectionError=pnp_retroprojection_error,
        confidence=pnp_confidence,
        flags=pnp_method,
    )
    # use Rodrigues formula (SO3) to obtain rotational matrix
    rotation_matrix, jacobian = cv2.Rodrigues(rotation_vector)
    # compute 3D pose in world coordinate
    translation_vector = -rotation_matrix.T @ translation_vector
    rotation_matrix = rotation_matrix.T
    # compute ratio and scale from the number of matched points and the current pose
    matched_point_ratio = len(idxPose) / len(keyframe_map)
    scale = np.linalg.norm(translation_vector)

    # project keyframe local map from camera pose OR update keyframe using the current frame / depth

    exit()
