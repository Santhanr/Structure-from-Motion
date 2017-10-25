'''http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_epipolar_geometry/py_epipolar_geometry.html
'''
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
'''

import numpy as np
import cv2

# Load the images in gray scale
img1 = cv2.imread('myleft.jpg', 0)
img2 = cv2.imread('myright.jpg', 0)

# Detect the SIFT key points and compute the descriptors for the two images
sift = cv2.xfeatures2d.SIFT_create()
keyPoints1, descriptors1 = sift.detectAndCompute(img1, None)
keyPoints2, descriptors2 = sift.detectAndCompute(img2, None)

# Create brute-force matcher object
bf = cv2.BFMatcher()

# Match the descriptors
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# Select the good matches using the ratio test
goodMatches = []
pts1=[]
pts2=[]

for m, n in matches:
    if m.distance < 0.8 * n.distance:
        goodMatches.append(m)
	pts2.append(keyPoints2[m.trainIdx].pt)
	pts1.append(keyPoints1[m.queryIdx].pt)

# Apply the homography transformation if we have enough good matches 
MIN_MATCH_COUNT = 10

allMatches=goodMatches
#goodMatches=goodMatches[:20]

if len(goodMatches) > MIN_MATCH_COUNT:
    # Get the good key points positions
    sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
    destinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1, 1, 2)
    
    # Obtain the homography matrix
    M, mask = cv2.findHomography(sourcePoints, destinationPoints, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    matchesMask = mask.ravel().tolist()
    print "Homography Matrix of Left & Right Image:"
    print M
    # Apply the perspective transformation to the source image corners
    h, w = img1.shape
    corners = np.float32([ [0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0] ]).reshape(-1, 1, 2)
    transformedCorners = cv2.perspectiveTransform(corners, M)
    
    # Draw a polygon on the second image joining the transformed corners
    img2 = cv2.polylines(img2, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(goodMatches), MIN_MATCH_COUNT))
    matchesMask = None

# Draw the matches
drawParameters = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask[:40], flags=2)
result = cv2.drawMatches(img1, keyPoints1, img2, keyPoints2, goodMatches[:40], None, **drawParameters)

# Display the results
cv2.imshow('Homography', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
pts1=np.float32(pts1)
pts2=np.float32(pts2)


pts_l = np.array(pts1)
pts_r = np.array(pts2)
K_l=np.array([[2760,0,1520],[0,2760,1006],[0,0,1]])
K_r=np.array([[2760,0,1520],[0,2760,1006],[0,0,1]])

print("Camera Matrix:")
print(K_l)
'''https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
'''
pts_l_norm = cv2.undistortPoints(np.expand_dims(pts_l, axis=1), cameraMatrix=K_l, distCoeffs=None)
pts_r_norm = cv2.undistortPoints(np.expand_dims(pts_r, axis=1), cameraMatrix=K_r, distCoeffs=None)

E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
print("Essential Matrix:")
print(E)

points, R, t, mask = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
print("Rotation Matrix:")
print(R)
print("Translation Matrix:")
print(t)
M_r = np.hstack((R, t))
M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

P_l = np.dot(K_l,  M_l)
P_r = np.dot(K_r,  M_r)
point_4d_hom = cv2.triangulatePoints(P_l, P_r, np.expand_dims(pts_l, axis=1), np.expand_dims(pts_r, axis=1))
point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
point_4d = point_4d[:3, :].T
fig = pyplot.figure()
ax = Axes3D(fig)
ax.autoscale(enable=True,axis='both',tight=None)
ax.scatter(point_4d[:,0], point_4d[:,1], point_4d[:,2])
pyplot.show()


