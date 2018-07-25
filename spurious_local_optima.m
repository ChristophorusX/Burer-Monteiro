clc
close all

n = 50;
theta = [1:n]'/n*2*pi;
%theta = 1.2*rand(n,1)*pi;
%theta(1) = 0;
%theta(n)= 1.5*pi;

%theta = zeros(n,1);
x = cos(theta);
y = sin(theta);
Q = [x y];
I = eye(n)

cvx_begin
cvx_precision low
variable A(n, n) symmetric
minimize -lambda_min(diag(diag(A * Q * Q')) - A.*(Q*Q'))
subject to
diag(A) == 0
A(:) + I(:) >= 1
% A(:) >= 1
A(:) <= sqrt(n)
diag(x) * (A * y) == diag(y) * (A * x)
cvx_end

L = diag(diag(A * Q * Q')) - A.*(Q*Q');
eigval = eig(L);
eigval(1:5)
