#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 2.7 and 2.8
# apply/use stationary wavelet

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pywt

def plot_coeffs(signal,coeffs):
	fig = plt.figure()
	N = len(coeffs)+2
	
	graph = fig.add_subplot(N,1,1)
	graph.plot(signal)
	graph.set_title('signal')
	
	for m in range(1,N-1):
		graph = fig.add_subplot(N,1,m+1)
		graph.plot(coeffs[m-1][1])
		graph.set_title('v_{:d}'.format(m))
	graph = fig.add_subplot(N,1,N)
	graph.plot(coeffs[N-3][0])
	graph.set_title('u_{:d}'.format(N-2))
	plt.show()

def main():
	# octave: y = zeros(128,1); y(28:78) = 1;
	x = np.zeros(128)
	for k in range(27,77):
		x[k] = 1.0
	
	wavelet = pywt.Wavelet('haar')
	coeffs = pywt.swt(x, wavelet, level=4)
	plot_coeffs(x,coeffs)

if __name__ == '__main__':
	main()
