import autocropper
import aligner
import roi_drawer
import feat_comparison
import matplotlib.pyplot as plt 
from skimage import io, img_as_ubyte
import os
import cv2

def make_template(impath, outpath, outtext):
	pic = io.imread(impath)
	cropped = autocropper.autocrop(pic)
	io.imsave(outpath, cropped)
	roi_drawer.draw_interface(outpath, outtext, scaling_factor=2)

def make_comparison(impath, cropped_path, template_path, roi_list, comp_name, mode="intensity"):
	pic = io.imread(impath)
	cropped = autocropper.autocrop(pic)
	template = cv2.imread(template_path)
	aligned = aligner.alignment_attempt(template, cropped)
	io.imsave(cropped_path, aligned)
	dig, gen, back = feat_comparison.import_rois(roi_list)
	I1 = cv2.imread(template_path)
	I2 = cv2.imread(cropped_path)
	if mode == "contour":
		comp = feat_comparison.compare_contour_props(I1,I2,gen, draw_mode=False)
	if mode == "intensity":
		comp = feat_comparison.compare_intensity(I1,I2,gen,back)
	io.imsave(comp_name, comp)
	plt.close("all")
	# fig, ax = plt.subplots()
	# ax.imshow(comp)
	# ax.axis("off")
	# plt.imshow(zic)
	# plt.show()


def cycle_over_files(relpath):
	for i, f in enumerate(os.listdir(relpath)):
		print f
		if f[-3:] == "JPG":
			print "working on {}".format(i)
			make_comparison(relpath + "/" + f, relpath + "/aligned{}.png".format(i), 
			relpath +"/template.png", relpath + "/rois.txt", relpath + "/areas{}.png".format(i), mode="contour")

relpath = "test_series2"
# make_template(relpath + "/GOOD.JPG", relpath + "/template.png", relpath + "/rois.txt")
cycle_over_files(relpath)


# missing:

# temp file hackk

# roi_digit_recogniser 
# chemtrails comparison
# (shading comparison)
# email