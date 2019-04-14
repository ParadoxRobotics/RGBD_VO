# NumPy/Python RGBD Visual Odometry for robotics applications:
# Author :  MUNCH Quentin 2018/2019

import numpy as np
import cv2
import imutils
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib import pyplot
import time

# Camera intrinsic parameters
fx =
fy =
cx =
cy =

# Initiate ORB object
orb = cv2.ORB_create(nfeatures=500, nlevels=3, scoreType=cv2.ORB_HARRIS_SCORE)
# Init feature matcher
matcher = cv2.DescriptorMatcher_create("BruteForce-L1")
# Init the D435 pipeline capture
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
# start D435
pipeline.start(config)

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
    matches = matcher.match(d_ref, d_cur)
    # filter match using SAD
    good_match = [m for m in matches if abs(kp_ref[m.queryIdx].pt[1] - kp_cur[m.trainIdx].pt[1])<3]
    # print matches
    img3 = cv2.drawMatches(ref_frame, kp_ref, cur_frame, kp_cur, good_match, None, flags=2)
    cv2.imshow('state', img3)
    print(cur_depth.shape)

    # create 2 points clouds

    # Inlier detection with RANSAC

    # Transformation estimation

    # update
    ref_frame = cur_frame
    ref_depth = cur_depth
    kp_ref = kp_cur
    d_ref = d_cur

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
pipeline.stop()
