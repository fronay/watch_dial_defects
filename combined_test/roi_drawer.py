""" # Example usage (as standalone):
python cv_quicklabel.py -i img.jpg -o dial_labels.txt
"""
import cv2
import argparse
import os

def draw_interface(image, output_file, scaling_factor):
    color_dict = {
        "general_roi": (0,255,0),
        "digit_roi": (255,0,0),
        "background": (0,0,255), 
        }
    # set mouse call back: 2-click-line
    def mark_bbox(event,x,y,flags,param):
        """mark 2 points and draw bounding rect"""
   
        global draw_mode, ref_point, region_mode
        region_color = color_dict[region_mode]
        if event == cv2.EVENT_LBUTTONDOWN and not draw_mode:
            cv2.circle(img, (x,y), 1, region_color, 1)
            ref_point[0] = (x,y)
            draw_mode = True
        elif event == cv2.EVENT_LBUTTONDOWN and draw_mode:
            cv2.circle(img, (x,y), 1, region_color, 1)
            ref_point[1] = (x,y)
            p1, p2 = ref_point[0], ref_point[1]
            cv2.rectangle(img, p1, p2, region_color, 1)
            append_to_roi(p1,p2, region_mode)
            draw_mode = False

    def append_to_roi(p1, p2, region_mode):
        (x1,y1), (x2,y2) = format_bbox(p1, p2)
        x1,y1,x2,y2 = [n*scaling_factor for n in (x1, y1, x2, y2)]
        img_roi.append("{},{},{},{},{}\n".format(x1, y1, x2, y2, region_mode))

    def format_bbox(p1, p2):
        """return bboxes in same format"""
        x1, y1 = p1
        x2, y2 = p2
        p1_new = (max(x1, x2), max(y1,y2))
        p2_new = (min(x1, x2), min(y1, y2))
        return p1_new, p2_new


    def setup_img(image, c_ratio):
        """load img, create clone, resize by conversion ratio for easy display"""
        if isinstance(image, basestring):
            print "working on", image
            img = cv2.imread(image)
        else:
            img = image.copy()
        img = cv2.resize(img, (0,0), fx=1/float(c_ratio), fy=1/float(c_ratio)) 
        clone = img.copy()
        return img, clone

    ####
    # points for rectangle:
    global draw_mode, ref_point, region_mode
    draw_mode = False
    ref_point = [(0,0), (0,0)]
    region_mode = "general_roi"
    # loop list img_roi for appending to global region list after img labeling end
    img_roi = []
    # conversion ratio for way-too-large images like my beloved TIFs
    # initialise img /clone before loop:
    img, clone = setup_img(image, scaling_factor)
    cv2.namedWindow("img", flags=cv2.WINDOW_NORMAL)

    # mouse callback function
    # keep looping until breaking with 'k'
    cv2.setMouseCallback("img", mark_bbox)

    while True:
        cv2.imshow("img", img)
        key = cv2.waitKey(1) & 0xFF
        # if 'r' key is pressed, reset image
        if key == ord("r"):
            img_roi = []
            img = clone.copy()
        elif key == ord("d"):
            region_mode = "digit_roi"
        elif key == ord("g"):
            region_mode = "general_roi"
        elif key == ord("b"):
            region_mode = "background"
        # if 'k' key pressed, cancel labeling for all images
        elif key == ord("k"):
            break

    # close all open windows
    cv2.destroyAllWindows()
    with open(output_file, "wb") as out:
        # then write one roi on each line
        out.write("".join(img_roi))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image folder")
    ap.add_argument("-o", "--output", required=True, help="Path to output file")
    args = vars(ap.parse_args())
    draw_interface(args["image"], args["output"], 3)

