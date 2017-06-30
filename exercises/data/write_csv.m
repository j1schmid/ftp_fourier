%
% MSE - FTP Fourier spring 2017
%
% Convert .mat files to csv file.
% NOTE: The python library scipy is able to read .mat files directly, see 
% ex06_denoising.py.
%
% ------------------------------------------------------------------------------
close all; clc; clear;
pkg load signal
% ------------------------------------------------------------------------------

load signals1.mat
dlmwrite ('signals1.csv', [u0' u1' u2' u3'], ',')

[x y] = pam_signal();
dlmwrite ('pam_signal.csv', [x y], ',')

% ------------------------------------------------------------------------------
