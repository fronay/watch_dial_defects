"""
Usage:
python aligner_test.py -t g.png -c g_rot.png -o "sample_alignment.png"

"""
import numpy as np
import cv2
import os
import argparse

# grayscale conversion for cv functions:
def gray(pic):
    return cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    Helper function for drawing image alignment feature matches (taken from opencv 3
    as not supported in cv 2.4)
    """
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype="uint8")

    # Place the first image to the left
    out[:rows1, :cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2, cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out

def calculate_sift_matches(img1, img2, min_matches=10, n_matches=100):
    """
    Take 2 images, calculate matching features and return comparison img + inverse transformation matrix
    """
    # Initiate SIFT detector
    sift = cv2.SIFT()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    MIN_MATCH_COUNT = min_matches
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    bf = cv2.BFMatcher()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # matches = bf.knnMatch(des1,des2, k=2)
    # store good matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good_matches.append(m)

    if len(good_matches)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        inverse_transformation = np.linalg.inv(M)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        tup = tuple(np.int32(dst)[1,0])
        # cv2.circle(img2, tup, 30, color=(0,255,0))
        # rectangle to visualise transform matrix
        ### 
        cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),5, cv2.CV_AA)
    else:
        print "Not enough matches are found - %d/%d" % (len(good_matches),MIN_MATCH_COUNT)
        matchesMask = None
        """
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)
        """
    # draw a max. of n_matches in the visualisation
    img3 = drawMatches(img1,kp1,img2,kp2,good_matches[:n_matches])
    return img3, inverse_transformation

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to the image at hand")
ap.add_argument("-c", "--comparison", required=True, help="Path to the image to be aligned")
ap.add_argument("-o", "--output", required=True, help="Name of aligned output image")
args = vars(ap.parse_args())

template = gray(cv2.imread(args["template"]))
comparison = cv2.imread(args["comparison"])

# try on single rotated image & ensemble image
# matched_rot_img, _ = calculate_sift_matches(img, img_rotated)
matched_ensemble_img, inv_m = calculate_sift_matches(template, gray(comparison.copy()))

target_shape = template.shape
recalibrated = cv2.warpPerspective(comparison, inv_m, target_shape)
print inv_m
#quick_fig(matched_ensemble_img)
cv2.imwrite("boo.png", matched_ensemble_img)
cv2.imwrite(args["output"], recalibrated)
