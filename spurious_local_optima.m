clc
close all

n = 20;
theta = [1:n]'/n*2*pi;
%theta = 1.2*rand(n,1)*pi;
%theta(1) = 0;
%theta(n)= 1.5*pi;

%theta = zeros(n,1);
x = cos(theta);
y = sin(theta);
Q = [x y];

cvx_begin
cvx_precision high
variable A(n, n) symmetric
minimize - lambda_min(diag(diag(A * Q * Q')) - A.*(Q*Q'))
subject to
A(:) >= 1
%            A(:) <= n
diag(x) * (A * y) == diag(y) * (A * x)
cvx_end

L = diag(diag(A * Q * Q')) - A.*(Q*Q');
eigval = eig(L);
eigval(1:5)
