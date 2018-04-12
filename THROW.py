from skimage import measure
from skimage.filters import try_all_threshold, threshold_triangle 
# from skimage.restoration import denoise_tv_chambolle
from skimage.morphology import erosion, opening, disk
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.color import label2rgb
from skimage import transform
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


print "completed imports"
elem = disk(5)
# orig = io.imread("Working_Copy/dail_photos/s3/ng/n0.JPG")
# orig = io.imread("Working_Copy/dail_photos/s3/g/IMG_9810.JPG")
# orig = io.imread("Working_Copy/dail_photos/s3/g/IMG_9816.JPG")
# orig = io.imread("Working_Copy/dail_photos/s2/ng/IMG_9802.JPG")
orig = cv2.imread("Working_Copy/dail_photos/s3/ng/IMG_9816.JPG")

print "completed load of image"
# resize (with scaling info to make thresholding easier afterwards)
def resize_with_scaling(img, single_factor=0.5, r_shape=None):
    orig_x, orig_y = img.shape[:2]
    if r_shape is not None:
        new_x, new_y = r_shape
        x_sf = orig_x*1.0/new_x
        y_sf = orig_y*1.0/new_y
    else:
        new_x = int(orig_x * 1.0 *single_factor)
        new_y = int(orig_y * 1.0 * single_factor)
        x_sf, y_sf = (single_factor, single_factor)
    resized = transform.resize(img, (new_x, new_y))
    return resized, (x_sf, y_sf)

def scale_label(label, scaling_factors):
    x,y = label
    x_sf, y_sf = scaling_factors
    x_new = int(orig_x*x_sf)
    y_new = int(orig_y*y_sf)
    return (x_new, y_new)

orig, scaling_factors = resize_with_scaling(orig, 0.1)
print orig.shape

print "completed resize"
# triangle threshold on lab space image (3rd channel :,:,2) works well for most types
orig_gr = rgb2lab(orig)
orig_gr = (orig_gr + [0, 128, 128]) / [100, 255, 255]
orig_gr = orig_gr[:,:,2]
thresh = threshold_triangle(orig_gr)
binary = orig_gr > thresh
print "completed thresholding"
# binary = opening(binary, elem)
binary = opening(binary, elem)
###

# remove artifacts connected to image border
cleared = clear_border(binary)

# label image regions
label_image = label(binary)
image_label_overlay = label2rgb(label_image, image=label_image)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)

print "completed opening/closing and stuff"   
# resize_with_labels(ima)

crop_bbox = None
for region in regionprops(label_image):
    # take regions with large enough areas 
    # usually 500*500 pixel on 1200*1920 images -> 400*400
    if region.area >= 1000*1000:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        crop_bbox = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        
if crop_bbox is not None:
    minr, minc, maxr, maxc = crop_bbox
    cropped_region = orig[minr:maxr, minc:maxc]
    quick_fig(cropped_region)
    # cropped_region = transform.resize(cropped_region, (510,510))
    # io.imsave("Working_Copy/dail_photos/s2/comp/template_hd.png", cropped_region)
ax.set_axis_off()
plt.tight_layout()
plt.show()