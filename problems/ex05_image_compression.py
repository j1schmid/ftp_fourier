#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 5.2 and 5.3
# apply the 2 dimensional discrete wavelet transform and thresholding to 
# compress an image

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pywt
from numpy import genfromtxt
import cv2
import os.path

def main():
	
	#bmp = cv2.imread(FILENAME + '.bmp', flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
	#jpg = cv2.imread(FILENAME + '.jpg', flags=cv2.CV_LOAD_IMAGE_GRAYSCALE)
	#png = cv2.imread(FILENAME + '.png') #, flags=cv2.COLOR_BGR2GRAY)
	#img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
	
	#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
	#cv2.imshow('image',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	MATLAB_IMG='data/Kueken.png'
	LENA_IMG='data/lena.bmp'
	
	if not os.path.isfile(MATLAB_IMG):
		print('\'{:s}.*\' not found, might get it from the fourier course folder, use \'{:s}\' instead.'.format(MATLAB_IMG,LENA_IMG))
		FILENAME = LENA_IMG
	else:
		FILENAME = MATLAB_IMG
	img = cv2.imread(FILENAME)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	#wavelet = pywt.Wavelet('coif1')
	#wavelet = pywt.Wavelet('db2')
	#wavelet = pywt.Wavelet('coif3')
	#wavelet = pywt.Wavelet('haar')
	wavelet = pywt.Wavelet('bior4.4')
	
	coeffs = pywt.dwt2(img,wavelet)
	
	top_ = np.concatenate((coeffs[0],coeffs[1][0]), axis=1)
	bottom_ = np.concatenate((coeffs[1][1],coeffs[1][2]), axis=1)
	dwt_img = np.concatenate((top_,bottom_), axis=0)
	
	fig = plt.figure()
	for k in range(1,5):
		graph = fig.add_subplot(2,2,k)
		if k == 1:
			img_tmp = coeffs[0]
		else:
			img_tmp = coeffs[1][k-2]
		graph.imshow(img_tmp, cmap = 'gray', interpolation = 'bicubic')
		#graph.xticks([]), graph.yticks([]) # to hide tick values on X and Y axis
	
	coeffs = pywt.wavedecn(img,wavelet)
	arr, coeff_slices = pywt.coeffs_to_array(coeffs)
	arr = pywt.threshold(arr, 2, 'hard')
	
	nb = arr.shape[0]*arr.shape[1]
	nnzb = np.count_nonzero(arr)
	
	coeffs = pywt.array_to_coeffs(arr, coeff_slices)
	img_rec = pywt.waverecn(coeffs,wavelet)
	
	mse = np.power(np.linalg.norm(img - img_rec,ord=2,axis=(0,1)),2)
	psnr = 10*np.log10(np.power(255,2)/mse)
	
	print(wavelet)
	print('level: {:d}'.format(len(coeffs)-1))
	print('{:d} none zero pixels of total {:d} pixels'.format(nnzb,nb))
	print('ratio {:1.3f}'.format(np.float(nnzb)/nb))
	print('MSE:  {:f}'.format(mse))
	print('PSNR: {:f}'.format(psnr))
	
	fig = plt.figure()
	graph = fig.add_subplot(1,2,1)
	graph.imshow(img, cmap = 'gray', interpolation = 'bicubic')
	graph = fig.add_subplot(1,2,2)
	graph.imshow(img_rec, cmap = 'gray', interpolation = 'bicubic')
	
	plt.show()
	
	#pdb.set_trace()

if __name__ == '__main__':
	main()
