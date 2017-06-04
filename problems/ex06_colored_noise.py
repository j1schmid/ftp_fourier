#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 6.3
# generate and visualize colored noise

from __future__ import print_function
import sys
import pdb

import numpy as np
import matplotlib.pyplot as plt

import scipy.signal


class pspectrum_plot:
	def __init__(self,figure_name):
		self.figure_name = figure_name
	
	def plot(self,signal,label):
	# compute and plot the power spectrum (128 bins) using welchs methode
		Pxx = scipy.signal.welch(signal, fs=1.0, window='hanning', nperseg=256, noverlap=128, nfft=None, detrend='constant', return_onesided=True, scaling='spectrum') # get power spectrum V**2/Hz
		
		plt.figure(self.figure_name) # select figure
		plt.plot(Pxx[0],10*np.log10(Pxx[1]),label=label)
	
	def plot_appendix(self):
		plt.legend(loc='best', borderaxespad=0.2)
		plt.xlabel('frequency / Hz')
		plt.ylabel('power density / dB/Hz')

def main():

	wnoise = np.random.randn(1024*16)
	# generate low-pass colored noise from the white noise
	lpnoise = scipy.signal.convolve(wnoise, [ 1,1], mode='same')
	hpnoise = scipy.signal.convolve(wnoise, [-1,1], mode='same')
	
	plt.figure('signals (time domain, only the first 128 samples)')
	plt.plot(wnoise[0:128],label='white noise')
	plt.plot(lpnoise[0:128],label='low-pass colored noise')
	plt.plot(hpnoise[0:128],label='high-pass colored noise')
	# just some plot beautyfing
	plt.xlabel('time / s')
	plt.ylabel('signal')
	plt.legend(loc='best', borderaxespad=0.2)
	
	power_spectrum = pspectrum_plot('power spectrum')
	power_spectrum.plot(wnoise,label='white noise')
	power_spectrum.plot(lpnoise,label='low-pass colored noise')
	power_spectrum.plot(hpnoise,label='high-pass colored noise')
	power_spectrum.plot_appendix()
	
	plt.show()

if __name__ == '__main__':
	main()
