# NumPy/Python RGBD Visual Odometry for robotics applications:
# Author :  MUNCH Quentin 2018/2019

"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2
import imutils
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from matplotlib import pyplot
import icp

# Camera intrinsic parameters
fx = 641.66
fy = 641.66
cx = 324.87
cy = 237.38

CIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

Rot_pose = np.eye(3)
Tr_pose = np.array([[0],[0],[0]])

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

    # create 2 points clouds with 5 random points pc = [x,y,z]
    ref_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_match])
    cur_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_match])
    pc_ref = np.zeros((len(ref_pts),3))
    pc_cur = np.zeros((len(cur_pts),3))

    for id in range(len(ref_pts)):
        # z component
        pc_ref[id, 2] = ref_depth[int(ref_pts[id, 1]), int(ref_pts[id, 0])]*depth_scale
        pc_cur[id, 2] = cur_depth[int(cur_pts[id, 1]), int(cur_pts[id, 0])]*depth_scale
        # x component
        pc_ref[id, 0] = (ref_pts[id, 0]-cx)*(pc_ref[id, 2]/fx)
        pc_cur[id, 0] = (cur_pts[id, 0]-cx)*(pc_cur[id, 2]/fx)
        # y component
        pc_ref[id, 1] = (ref_pts[id, 1]-cy)*(pc_ref[id, 2]/fy)
        pc_cur[id, 1] = (cur_pts[id, 1]-cy)*(pc_cur[id, 2]/fy)

    # ICP
    T, distances, iterations = icp.icp(pc_cur, pc_ref, tolerance=0.000001)
    ROT = T[0:3, 0:3]
    TR = np.array([[T[0,3]],[T[1,3]],[T[2,3]]])

    # trajectory calculation
    Rot_pose = np.dot(ROT, Rot_pose) 
    Tr_pose = Tr_pose + np.dot(Rot_pose, TR)

    # update
    ref_frame = cur_frame
    ref_depth = cur_depth
    kp_ref = kp_cur
    d_ref = d_cur

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
pipeline.stop()
