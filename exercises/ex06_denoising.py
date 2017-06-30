#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 6.1 and 6.2
# denoise signals by thresholding the discrete wavelet transform and reconstruct
# the signal, compare soft and hard thresholding, determine the threshold
# automaticaly

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pywt
from numpy import genfromtxt
import cv2
import scipy.io
import os.path
from numpy import genfromtxt

def denoise(data,K,wname,level,threshold_type):
	wavelet = pywt.Wavelet(wname)
	if level=='None':
		coeffs = pywt.wavedecn(data,wavelet)
	else:
		coeffs = pywt.wavedecn(data,wavelet,level=level)
	# estimate the standard deviation of the noise using the finest detail / nu
	# coefficients
	std_dev = np.median(np.abs(coeffs[-2]['d'])) / 0.6745
	# compute the threshold according to the MSE FTP_Fourier slides week 13
	threshold = K * np.sqrt(2*np.log(len(data))) * std_dev	
	arr, coeff_slices = pywt.coeffs_to_array(coeffs)
	arr = pywt.threshold(arr, threshold, threshold_type)
	coeffs = pywt.array_to_coeffs(arr, coeff_slices)
	return (threshold, pywt.waverecn(coeffs,wavelet))

def show_denoised(data,K,wname,level='None'):
	threshold, dn_hard = denoise(data,K,wname,level,'hard')
	_, dn_soft = denoise(data,K,wname,level,'soft')
	print('threshold: {:f}'.format(threshold))
	
	plt.plot(dn_soft,label='\''+wname+'\' soft threshold')
	plt.plot(dn_hard,label='\''+wname+'\' hard threshold')
	return threshold

def main():
	fig = plt.figure()
	
	FILENAME='data/EEG1.MAT'
	if os.path.isfile(FILENAME):
		print('use file \'{:s}\' from fourier course'.format(FILENAME))
		mat = scipy.io.loadmat(FILENAME)
		meas = mat['sig'][0]
	else:
		FILENAME2='data/pam_signal.csv'
		print('file \'{:s}\' from fourier course not found, use \'{:s}\' instead'.format(FILENAME,FILENAME2))
		
		data = genfromtxt(FILENAME2, delimiter=',')
		meas = data[:,1]
		plt.plot(data[:,0],label='original') 
		print('signal length is {:d}'.format(len(meas)))
		
	plt.plot(meas,label='measured signal')
	threshold = show_denoised(meas,1.0,'haar')
	#show_denoised(meas,1.0,'bior4.4',8)
	
	plt.legend(loc='best', borderaxespad=0.2)
	plt.show()

	#pdb.set_trace()

if __name__ == '__main__':
	main()
