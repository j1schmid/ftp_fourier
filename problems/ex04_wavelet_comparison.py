#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

# mse ftp_fourier (spring semester 2017)
#
# application exercise 4.2 and 4.3
# compare some wavelets and its features

from __future__ import print_function
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pywt

def main():
	
	# pywt.wavelist()
	wavelet_names = ['haar','db2','sym2','sym5','coif1','bior1.3','bior2.2']
	#wavelet_names = ['bior1.3','bior2.2']
	
	for wavelet_name in wavelet_names:
		
		N=4
		wavelet = pywt.Wavelet(wavelet_name)
		
		n = len(wavelet.filter_bank[0])
		
		print('wavelet name: \'{:s}\''.format(wavelet.name))
		print(wavelet)
		print('Vanishing moments phi: {:d}'.format(wavelet.vanishing_moments_phi))
		print('                  psi: {:d}'.format(wavelet.vanishing_moments_psi))
		
		start = np.zeros(n+1+4)
		start[n/2+2] = 1
		
		#psi = pywt.idwt(None,[0,0,1,0,0], wavelet,mode='zero')
		psi = pywt.idwt(None,start, wavelet,mode='zero')
		for k in range(0,N-1):
			psi = pywt.idwt(psi,None, wavelet,mode='zero')
		
		phi = pywt.idwt(start,None, wavelet,mode='zero')
		for k in range(0,N-1):
			phi = pywt.idwt(phi,None, wavelet,mode='zero')
		
		fig = plt.figure()
		graph = fig.add_subplot(111)
		graph.plot(psi)
		graph.plot(phi)

		#[phi, psi, x] = wavelet.wavefun(N)
		#graph.plot(psi)
		#graph.plot(phi)

		plt.show()
		
	#pdb.set_trace()

if __name__ == '__main__':
	main()
