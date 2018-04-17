import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage import io
from skimage.filters import threshold_mean, gaussian
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb, rgb2gray, rgb2lab
import matplotlib.patches as mpatches


# https://docs.opencv.org/3.4.1/d1/d32/tutorial_py_contour_properties.html

with open("test_roi_drawer/fugu.txt","r") as readfile:
    dig = []; gen = []; back = [];
    for line in readfile.readlines():
        lit = line.replace("\n","").split(",")
        vals = [int(n) for n in lit[:-1]]
        if lit[-1] == "digit_roi":
            dig.append(vals)
        elif lit[-1] == "general_roi":
            gen.append(vals)
        elif lit[-1] == "background":
            back.append(vals)
            
for i in range(11): #[1,3]:    range(12):
    n1 = io.imread("roman/download-{}.jpg".format(i))
    n2 = io.imread("roman/alternative-{}.jpg".format(i))
    #n1 = io.imread("roman/marks/download-{}.jpeg".format(i))
    #n2 = io.imread("roman/marks/a-{}.jpeg".format(i))
    #n1 = io.imread("roman/numbers/download-{}.jpeg".format(i))
    #n2 = io.imread("roman/numbers/alternative-{}.jpeg".format(i))
    def show_contours(sic):
        # quick_fig(sic)
        pic = rgb2gray(sic.copy())
        #pic = (pic + [0, 128, 128]) / [100, 255, 255]
        #pic = pic[:,:,2]
        fig, ax = plt.subplots()
        ax.imshow(pic, interpolation='nearest', cmap=plt.cm.gray)
        contours = measure.find_contours(pic, 0.1)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
        return contours

    c1 = show_contours(n1)
    c2 = show_contours(n2)
    """
    def convert_contours(contours):
        converted = []
        # mask = np.zeros_like(toral)
        for n, contour in enumerate(contours):
            # append contour with empty dimension for cv2 compatibility
            c_conv = np.expand_dims(np.fliplr(contour), axis=1).astype(np.int32)
            if cv2.contourArea(c_conv) > 6000:
                converted.append(c_conv)
        return converted

    conv1 = convert_contours(c1)
    conv2 = convert_contours(c2)

    def show_contour_props(cnt):
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        aspect_ratio = float(MA)/ma
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        return [area, aspect_ratio, angle, solidity]

    props1 = [show_contour_props(c) for c in conv1]
    props2 = [show_contour_props(c) for c in conv2]

    print "{}, area:{}, aspect ratio:{}, angle:{}, solidity: {}".format(i, *props1[0])
    print "{}, area:{}, aspect ratio:{}, angle:{}, solidity: {}".format(i, *props2[0])
    # deltas = []
    # for g in range(len(props1[0])):
        # print 1.0*props1[0][g] - props2[0][g]#  / 1.0*props1[0][g]
        # print props2[0][g] # 1.0*props1[0][g]
    # print "d_ar:{}, d_aspect ratio:{}, d_angle:{}, d_solidity: {}".format(*deltas)
    """
    plt.show()
    # print props2
    
# plt.show()
