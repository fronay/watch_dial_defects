import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import measure
from skimage import io
from skimage.filters import threshold_mean, gaussian
from skimage.util import img_as_ubyte
from skimage.color import gray2rgb, rgb2gray, rgb2lab
import matplotlib.patches as mpatches
 

def import_rois(path):
    with open(path,"r") as readfile:
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
    return dig, gen, back

def convert_contours(contours, min_area):
        converted = []
        # mask = np.zeros_like(toral)
        for n, contour in enumerate(contours):
            # append contour with empty dimension for cv2 compatibility
            c_conv = np.expand_dims(np.fliplr(contour), axis=1).astype(np.int32)
            if cv2.contourArea(c_conv) > min_area:
                converted.append(c_conv)
        return converted

def get_contours(sub_roi, contour_level=0.1, min_area_fraction=0.07, show_drawing=False):
    roi_area = sub_roi.shape[0]*sub_roi.shape[1]
    min_area = min_area_fraction*roi_area
    pic = rgb2gray(sub_roi.copy())
    contours = measure.find_contours(pic, contour_level)
    cv_cnts = convert_contours(contours, min_area)
    if len(cv_cnts) == 0 and contour_level < 1.0:
        # print "recursive search of better contour level started"
        contour_level += 0.1
        cv_cnts = get_contours(sub_roi, contour_level, min_area, show_drawing=False)
    elif len(cv_cnts) == 0 and contour_level >= 1.0:
        print "could not find any suitable contours at any level"
        return None
    if show_drawing:
        fig, ax = plt.subplots()
        ax.imshow(pic, interpolation='nearest', cmap=plt.cm.gray)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        ax.axis('image')
        ax.set_xticks([])
        ax.set_yticks([])
    return cv_cnts

def show_contour_props(cnt):
    area = cv2.contourArea(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
    aspect_ratio = float(MA)/ma
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    return [area, aspect_ratio, angle, solidity]

def cycle_over_contour_props(image, roi_list, draw_mode=True):
    img = image.copy()
    all_props = []
    for r in roi_list:
        x2,y2,x1,y1 = r
        sub_roi = img[y1:y2,x1:x2]
        cnts = get_contours(sub_roi)
        if cnts is not None:
            props = [show_contour_props(c) for c in cnts]
            largest_prop = max(props, key=lambda x: x[0])
            # print "number of large contours found:", len(props)
            # print "area:{}, aspect ratio:{}, angle:{}, solidity: {}".format(*largest_prop)
            all_props.append(largest_prop)
            if draw_mode:
                _ = [cv2.drawContours(sub_roi,c,-1,(0,255,0),3) for c in cnts]
        else:
            all_props.append([None,None,None,None])
    if draw_mode:
        plt.imshow(img)
    return all_props

def change_fraction(x,y):
    return ((float(x)-y)/x)

def compare_contour_props(img1,img2,roi_list, draw_mode=False):
    print "running contour comparison..."
    cp1 = cycle_over_contour_props(img1,roi_list)
    cp2 = cycle_over_contour_props(img2,roi_list)
    sol1 = [i[3] for i in cp1]
    sol2 = [i[3] for i in cp2]
    arat1 = [i[1] for i in cp1]
    arat2 = [i[1] for i in cp2]
    clone = img2.copy()
    for i in range(len(roi_list)):
        x2,y2,x1,y1 = roi_list[i]
        if None in cp1[i]:
            print "product defect or failed contour detection found"
            # cv2.rectangle(img2, (x1,y1), (x2,y2), (0,0,255), 2)
        elif None in cp2[i]:
            print "product defect or failed contour detection found"
            cv2.rectangle(clone, (x1,y1), (x2,y2), (255,0,0), 2)
        elif change_fraction(sol1[i], sol2[i]) > 0.2 and change_fraction(arat1[i], arat2[i]) > 0.2:
            print "sol1, sol2, arat1, arat2", sol1[i], sol2[i], arat1[i], arat2[i]
            print "large disparity found - solidity {} or aspect ratio {}".format(change_fraction(sol1[i], sol2[i]), change_fraction(arat1[i], arat2[i]))
            cv2.rectangle(clone, (x1,y1), (x2,y2), (255,0,0), 2)
        else:
            "print all good"
            cv2.rectangle(clone, (x1,y1), (x2,y2), (0,255,0), 2) 
    if draw_mode: 
        print "plotting..."
        plt.axis("off")
        plt.imshow(clone)
    return clone

def compute_thresholds(input_img,background_rois):
    threshes = []
    for i in range(0, len(background_rois)):
        x2,y2,x1,y1 = background_rois[i]
        thresh = np.mean(input_img[y1:y2,x1:x2])
        threshes.append(thresh)
    thresh_max = max(threshes)
    thresh_min = min(threshes)
    return thresh_max, thresh_min

def thresh_roi(img, bbox, tmax, tmin):
    x2,y2,x1,y1 = bbox
    roi = img[y1:y2,x1:x2]
    roi[roi>1.15*tmax]=255
    roi[roi<0.85*tmin]=0
    return roi

def compare_intensity(img1,img2,roi_list, background_rois):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    print "running intensity comparison..."
    tmax, tmin = compute_thresholds(gray1, background_rois)
    differences = []
    for i in range(len(roi_list)):
        x2,y2,x1,y1 = roi_list[i]
        sub_roi1 = thresh_roi(gray1,roi_list[i],tmax,tmin)
        sub_roi2 = thresh_roi(gray2,roi_list[i],tmax,tmin)
        good_value = np.mean(sub_roi1)
        suspect_value = np.mean(sub_roi2)
        difference = change_fraction(good_value, suspect_value)
        differences.append(difference)
        if difference > 0.15:
            print "found significant difference"
            cv2.rectangle(gray2, (x1,y1), (x2, y2), 0, 2)
        else: 
            cv2.rectangle(gray2, (x1,y1), (x2, y2), 255, 2)
    print "avg difference: ", sum(differences)/len(differences)
    return gray2



if __name__ == "__main__":
    dig, gen, back = import_rois("template.txt")
    im1 = cv2.imread("template.png")
    im2 = cv2.imread("comparison.png")
    compare_contour_props(im1, im2, gen, True)
    plt.show()
