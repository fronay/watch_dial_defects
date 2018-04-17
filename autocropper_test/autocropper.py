"""
Usage (if standalone):
python autocropper.py -i IMG_9814.JPG -o cropped.png
"""
from skimage import measure
from skimage.filters import threshold_triangle 
from skimage.morphology import erosion, opening, disk
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2lab
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage import io
import matplotlib.patches as mpatches
import argparse


def quick_fig(img):
    # quick figure utility function using matplotlib
    plt.figure(figsize=(16,5)) 
    plt.imshow(img)
    plt.axis("off")


def resize_with_scaling(img, single_factor=0.5, r_shape=None):
    # resize (with scaling info to make thresholding easier afterwards)
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
    x_new = int(x*1.0/x_sf)
    y_new = int(y*1.0/y_sf)
    return (x_new, y_new)

def autocrop(image, scale=0.05, show_plot=False, save_name=None):
    orig, scaling_factors = resize_with_scaling(image, scale)
    print "completed resize"

    struct_elem_size = max(5,int(0.005*orig.shape[0]))
    print struct_elem_size
    elem = disk(struct_elem_size)
    orig_gr = rgb2lab(orig)
    orig_gr = (orig_gr + [0, 128, 128]) / [100, 255, 255]
    im_means = []
    threshed_images = []
    for lab_channel in range(3):
        working_copy = orig_gr.copy()[:,:,lab_channel]
        thresh = threshold_triangle(working_copy)
        binary = working_copy > thresh
        threshed_images.append(binary)
        # quick_fig(binary)
        print np.mean(binary)
        im_means.append(np.mean(binary))
        print "completed thresholding for channel {}".format(lab_channel)
        # min_mean_channel = im_means.index(min(im_means))
        # binary = threshed_images[min_mean_channel]

        # label image regions
        label_image = label(binary)
        image_label_overlay = label2rgb(label_image, image=label_image)

        crop_bbox = None
        overall_area = label_image.shape[:2][0]*label_image.shape[:2][1]
        for region in regionprops(label_image):
            # print overall_area, 0.05*overall_area
            if 0.75*overall_area > region.area >= 0.07*overall_area:
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                crop_bbox = region.bbox
                # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
                # ax.add_patch(rect)
                break

    quick_fig(np.hstack(threshed_images))
    if crop_bbox is not None:
        minr, minc, maxr, maxc = crop_bbox
        minr, minc = scale_label((minr, minc), (scale, scale))
        maxr, maxc = scale_label((maxr,maxc), (scale, scale))
        cropped_region = image[minr:maxr, minc:maxc]
        quick_fig(cropped_region)
        print cropped_region.shape
        if cropped_region.shape[0]>1000:
            cropped_region = transform.resize(cropped_region, (1000,1000))
        else:
            cropped_region = transform.resize(cropped_region, (500,500))
        if save_name is not None:
            print "saving cropped image as {}".format(save_name)
            io.imsave(save_name, cropped_region)
        else:
            return cropped_region

    if show_plot:
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image at hand")
    ap.add_argument("-o", "--output", required=True, help="Name of output image name")
    args = vars(ap.parse_args())
    image = io.imread(args["image"])
    autocrop(image, scale=0.05, show_plot=True, save_name=args["output"])
