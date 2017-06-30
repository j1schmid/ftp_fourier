% 
% MSE - Fourier FS17
%
% Week 2
%
% JSCH 2017-03-04
% ------------------------------------------------------------------------------
close all; clc; clear;
pkg load signal
format compact short 
% ------------------------------------------------------------------------------


%%
printf ('\n')
printf ('1 orthonormal basis in R^3\n')
printf ('--------------------------\n')
%

v1 = [3 4 12]'
v2 = [12 3 -4]'

printf('a) v1^T * v2 = %f\n',dot(v1,v2))
% b)
e1 = v1/norm(v1,2) % = v1/norm(v1)
e2 = v2/norm(v2)
% c)
e3 = cross(e1,e2)

v = [1 2 3]'
% d)
alpha = dot(v,e1)
beta = dot(v,e2)
gamma = dot(v,e3)

% e)
norm(v)^2
alpha^2+beta^2+gamma^2

% ------------------------------------------------------------------------------

%%
printf ('\n')
printf ('2 dual basis\n')
printf ('------------\n')
%

b1 = [1 1 2]'; b2 = [2 2 3]'; b3 = [3 1 1]';

dot(b1,b2), dot(b1,b3), dot(b2,b3)
b1'*cross(b1,b2)
b2'*cross(b1,b3)
b3'*cross(b1,b2)

D = ([b1 b2 b3]^-1)'

d1 = D(:,1)
d2 = D(:,2)
d3 = D(:,3)

% ------------------------------------------------------------------------------

%%
% 3 Space of polynomials
%

x = linspace(-1,1,1e3);

% set up basis
p_b(1,:) = sqrt(1/2) * ones(size(x));
p_b(2,:) = sqrt(3/2) * x;
p_b(3,:) = sqrt(5/8) * polyval([-3  0  1],x); % (1 - 3*x.^2);
p_b(4,:) = sqrt(7/8) * polyval([-5  0  3  0],x); % (3x - 5*x.^3);

% polynom to decompose
p = polyval([1 1 1 1],x);

c = [sqrt(2)*4/3   sqrt(2/3)*8/5  -sqrt(5*8)/15  -sqrt(7/8)*8/35];
p_rec = c * p_b;

figure('Name','P3 Space of polynomials')
	plot(x,p_b  ,  x,p  ,  x,p_rec,'--')
	legend('p_1(x) = sqrt(1/2)','p_2(x) = sqrt(3/2)*x','p_3(x) = sqrt(5/8)*(1 - 3*x.^2)','p_4(x) = sqrt(7/8)*(3x - 5*x.^3)','p(x) = 1 + x + x^2 + x^3','reconstructed p(x)')

% ------------------------------------------------------------------------------

%%
% 4 Fourier Series
%

Ts = 1e-4;
x = 0:Ts:2*pi;
f = pi-x;

g0 = ones(size(f))/sqrt(2*pi);
c0 = sum(g0.*f)*Ts;

f_hat = c0*g0;

figure
%      plot(x,g0), hold on

for k=1:50
    g_sin(k,:) = sin(k*x)/sqrt(pi);
    g_cos(k,:) = cos(k*x)/sqrt(pi);
%      plot(x,g_sin(k,:) , x,g_cos(k,:))    
    c_sin(k) = sum(g_sin(k,:).*f)*Ts;
    c_cos(k) = sum(g_cos(k,:).*f)*Ts;

    f_hat = f_hat + c_sin(k)*g_sin(k,:) + c_cos(k)*g_cos(k,:);
    
end % for

plot(x,f  ,  x,f_hat)


% analytical
x = linspace(0,2*pi,1e4);
f = pi-x;

plot(x,f,'g'), hold on
for n=10:4:50
    f_hat = zeros(size(x));
    for k=1:n
        c_sin = sqrt(pi)*2/k;
        f_hat = f_hat + c_sin*sin(x*k)/sqrt(pi);
    end % for
    plot(x,f_hat)
end % for

% ------------------------------------------------------------------------------

%%
printf ('\n')
printf ('5 Matlab\n')
printf ('--------\n')
%

% a)

N = 10;
x = rand(N,1) + j*rand(N,1); y = rand(N,1) + j*rand(N,1);
dot(x,y)

% b)

n = 5; s = 1/sqrt(5);
A = s*fft(eye(n)) % fft of each column of exe(...)
printf('rank of A = %f, determinante of A = %f\n',rank(A),det(A))
printf('scalar product of colums of A to itself:\n')
for cnt=1:n
    printf('%f ',A(:,cnt)'*A(:,cnt))
end
printf('\n')
printf('or look at the diagonal of A^T*A')
A'*A

% c)
m = 5; n = 4; printf('A is a m x n = %d x %d = (height x width) matrix\n',m,n)
d = rand(n,1); d = 2*ones(n,1)
A = rand(m,n)
ones(m,1)*d'
A = A.*(ones(m,1)*d')
A = A*diag(d)
printf('A(1,4) = %f\n',A(1,4))

d = A'*A.*eye(n)
d = sqrt(inv(d))
d = diag(d) % get the diagonal of A as a column vector
A = A*diag(d)

printf('A^T*A:\n')
A'*A

B = rand(n,n)
D = (B^-1)'

printf('B^T*D:\n')
B'*D

% ------------------------------------------------------------------------------
