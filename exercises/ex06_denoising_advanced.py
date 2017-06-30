#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 6.4
# estimate the "power" of noise in different levels of a wavelet decomposition

from __future__ import print_function
import sys
import pdb

import numpy as np
import matplotlib.pyplot as plt
import pywt

import scipy.signal

def estimate_sigma(coeffs,level):
	return np.median(np.abs(coeffs[level+1]['d'])) / 0.6745

def estimate_sigmas(signal, wavelet):
	coeffs = pywt.wavedecn(signal,wavelet)
	N = len(coeffs)-1
	sigmas = []
	for m in range(0,N-1):
		sigmas.append(estimate_sigma(coeffs,m))
	return sigmas

def determine_threshold(K,std_dev,signal):
	# signal might be the coefficients of a single level
	return K * np.sqrt(2*np.log(len(signal))) * std_dev

def denoise_level_by_level(signal,K,wavelet,sigmas,threshold_type):
	N = len(sigmas)-1
	coeffs = pywt.wavedecn(signal,wavelet,level=N)
	
	coeffs[0] = pywt.threshold(coeffs[0],sigmas[0]) # approximation coefficients TODO
	for m in range(0,N):
		coeffs[m+1]['d'] = pywt.threshold(coeffs[m+1]['d'],sigmas[m],mode=threshold_type)
	return pywt.waverecn(coeffs,wavelet)

def main():

	wnoise = np.random.randn(1024)
	# generate low-pass colored noise from the white noise
	lpnoise = scipy.signal.convolve(wnoise, [ 1,1], mode='same')
	hpnoise = scipy.signal.convolve(wnoise, [-1,1], mode='same')
	
	wavelet = pywt.Wavelet('db2')
	
	w_sigmas = estimate_sigmas(wnoise,wavelet)
	lp_sigmas = estimate_sigmas(lpnoise,wavelet)
	hp_sigmas = estimate_sigmas(hpnoise,wavelet)
	
	plt.figure('std. deviations of the wavelet decomposition')
	plt.plot(w_sigmas,label='white noise')
	plt.plot(lp_sigmas,label='low-pass colored noise')
	plt.plot(hp_sigmas,label='high-pass colored noise')
	# just some plot beautyfing
	plt.legend(loc='best', borderaxespad=0.2)
	plt.xlabel('level')
	plt.ylabel('standard deviation')
	
	noise = lpnoise
	sigmas = lp_sigmas
	
	time = np.linspace(.0, 3.0, num=len(noise))
	signal = 20*np.abs(np.sin(time*2*np.pi))
	est_local = denoise_level_by_level(signal+noise,2.0,wavelet,sigmas,'soft')
	
	plt.figure('denoising')
	plt.plot(noise+signal,label='noisy signal')
	plt.plot(signal,label='original signal')
	plt.plot(est_local,label='level by level noise rejected')
	
	show_denoised(signal+noise,2.0,'db2',level='None')
	
	plt.legend(loc='best', borderaxespad=0.2)
	
	plt.show()
	#pdb.set_trace()

# ------------------------------------------------------------------------------
# global thresholding (copy from ex06_denoising.py)
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
# ------------------------------------------------------------------------------

if __name__ == '__main__':
	main()
