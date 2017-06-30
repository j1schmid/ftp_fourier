function [x,y] = pam_signal(N = 1024, Pnoise = 1)
% ------------------------------------------------------------------------------
% MSE - Fourier FS17
%
% Signal Generator, just to use as an example for wavelet decomposition
%
% [x,y] = pam_signal; plot(x,'-',y,'-')
%
% JSCH 2017-05-08
% ------------------------------------------------------------------------------

    %  nPAM = 8;
    %  round(nPAM*rand(N,1));
    xseq = [0 0 3 2 2 1 1 1 1 -1 -1 -1 -1 -1 2 2 2 -3 -3 2 1 1 1 1 1 0 0 0 0 -2 -2 -2]';
    L = N/length(xseq);
    x = upsample(xseq,L);
    x = conv(x,ones(L,1)); x = x(1:N);
    y = x + sqrt(Pnoise)*randn(size(x));
%      plot(x,'-',y,'-')
% ------------------------------------------------------------------------------
end % function
