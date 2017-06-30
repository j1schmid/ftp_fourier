#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 4.3 and 4.4
# reconstruct a signal from a subset of the coefficients, i.e. skip/ignore the
# fine scale coefficients
#
# ATTENTION
# I don't know why the overlengt (signal length not equal to a 2^n or other
# wavelets than the haar) occurs and how to interpret it.

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pdb
import pywt
from numpy import genfromtxt
import os.path

def dwt_rec_plot(signal,coeffs,wavelet,num_of_levels=-1):
	# reconstruct the signal using only a subset of the coefficients, of course,
	# using a orthogonal (not biorthogonal) wavelet this are equal to the 
	# approximation coefficients/signal of the decomposition
	
	N = len(coeffs) - 1 # determine decomposition level
	
	#plt.plot(signal,label='original signal')
	
	x = pywt.waverec(coeffs, wavelet)
	plt.plot(x,label='fully reconstructed signal')
	
	print('signal (len={:d}) is decomposed in {:d} levels/resolutions'.format(len(signal),N)) # plot the number of levels
	print('fully reconstruction is of length {:d}'.format(len(x)))
	print(' level | len(approx) | len(x)')
	for m in range(N-1,num_of_levels,-1):
		approx = pywt.waverec(coeffs[0:m+1], wavelet)
		# approx = np.asarray(approx) still is an array
		x = approx.repeat(np.power(2,N-m),axis=0)
		plt.plot(x,label='up to v_{:d}'.format(m))
		print('    {:2d} |         {:3d} | {:3d}'.format(m,len(approx),len(x)))
	
	plt.legend(loc='best', borderaxespad=0.2)
	
	# mark the sampling positions
	x = pywt.waverec(coeffs, wavelet)
	plt.plot(x,linestyle='None',marker='x', color='k', markersize=5)
	for m in range(N-1,num_of_levels,-1):
		x = pywt.waverec(coeffs[0:m+1], wavelet)
		t = range(0,len(x)*np.power(2,N-m),np.power(2,N-m))
		plt.plot(t,x,linestyle='None', marker='x', color='k', markersize=5)
		
# a previouse version of the dwt_rec_plot function using subplots
#def dwt_rec_plot(signal,coeffs,wavelet):
	#fig = plt.figure()
	#N = len(coeffs)
	
	#graph = fig.add_subplot(N-1,1,1)
	#graph.plot(signal)
	#graph.plot(pywt.waverec(coeffs, wavelet))
	#graph.set_title('original signal & fully reconstructed signal')

	#for m in range(2,N):
		#graph = fig.add_subplot(N-1,1,m)
		#graph.plot(pywt.waverec(coeffs[0:m], wavelet))
		#graph.set_title('up to v_{:d}'.format(m))
	
	#print('m = {:d}, len(x) = {:d}, len(approx) = {:d}'.format(m,len(x),len(approx)))

def main():
	FILENAME='data/signals1.csv'
	if os.path.isfile(FILENAME):
		my_data = genfromtxt(FILENAME, delimiter=',')
		x = my_data[:,0] # rectangular
		x = my_data[:,1] # sin(t) inclusiv jump
		x = my_data[:,2] # sin(t) diff. freq. and jump
		x = my_data[:,3] # rect, sin
		x = x[20:20+128]
	else:
		n = np.linspace(0,1.2*np.pi,128)
		x = np.sin(n*2)
	
	wavelet = pywt.Wavelet('haar')
	
	[phi, psi, _] = wavelet.wavefun(5)
	fig = plt.figure()
	plt.plot(psi)
	plt.plot(phi)
	
	
	fig = plt.figure()
	coeffs = pywt.wavedec(x, wavelet)
	#coeffs = pywt.wavedec(x, wavelet, level=4)
	dwt_rec_plot(x,coeffs,wavelet)
	#plot_dwt_rec_old(x,coeffs,wavelet)
	plt.show()
	
	#pdb.set_trace()

if __name__ == '__main__':
	main()
