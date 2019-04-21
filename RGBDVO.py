# NumPy/Python RGBD Visual Odometry for robotics applications:
# Author :  MUNCH Quentin 2018/2019

import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import imutils
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib import pyplot
import time

# Camera intrinsic parameters
fx = 641.66
fy = 641.66
cx = 324.87
cy = 237.38

CIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000, nlevels=8, scoreType=cv2.ORB_FAST_SCORE)
# Init feature matcher
matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
# Init the D435 pipeline capture
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
# start D435 and recover scale
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

# alignement object
align_to = rs.stream.color
align = rs.align(align_to)

# Acquire state
ref_state = pipeline.wait_for_frames()
# Align the depth frame to color frame
ref_aligned_state = align.process(ref_state)
ref_frame_aligned = ref_aligned_state.get_color_frame()
ref_depth_aligned = ref_aligned_state.get_depth_frame()
# get reference RGB state
ref_frame = np.asanyarray(ref_frame_aligned.get_data())
ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
# get reference Depth state
ref_depth = np.asanyarray(ref_depth_aligned.get_data())

# find keypoints
kp_ref, d_ref = orb.detectAndCompute(ref_frame, None)

while True:
    # Acquire new state
    cur_state = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    cur_aligned_state = align.process(cur_state)
    cur_frame_aligned = cur_aligned_state.get_color_frame()
    cur_depth_aligned = cur_aligned_state.get_depth_frame()
    # new RGB state
    cur_frame = np.asanyarray(cur_frame_aligned.get_data())
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    # new Depth state
    cur_depth = np.asanyarray(cur_depth_aligned.get_data())

    # find keypoints
    kp_cur, d_cur = orb.detectAndCompute(cur_frame, None)
    # make match
    matches = matcher.knnMatch(d_ref, d_cur, 2)
    # filter match using lowe loss
    good_match = []
    good_match_print = [] # only fo debug
    for m,n in matches:
        if m.distance < 0.45*n.distance:
            good_match.append(m)
            good_match_print.append([m])
    # print matches
    img3 = cv2.drawMatchesKnn(ref_frame, kp_ref, cur_frame, kp_cur, good_match_print, None, flags=2)
    cv2.imshow('state', img3)

    # find the 5 best inlier with ransac
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_match])
    cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_match])

    print(cur_pts.shape)

    # create 2 points clouds with 5  best points pc = [x,y,z]
    pc_ref = np.zeros((5,3))
    pc_cur = np.zeros((5,3))

    # local trajectory optimization (5 last pose + keypoint)

    # update
    ref_frame = cur_frame
    ref_depth = cur_depth
    kp_ref = kp_cur
    d_ref = d_cur

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
pipeline.stop()
