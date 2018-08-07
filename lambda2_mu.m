clear all
clc
close all
Vec = @(X) X(:);

%% Plot
n = 1000;
j_ind = [1:n]';
lambda_k = 2*cos(j_ind*pi*2/n) - 2*(cos(j_ind*pi*2/n)).^2;
cumsumlam = cumsum(lambda_k)/n;
plot(2*[1:n]/n,cumsum(lambda_k)/n, 'color', [0.402 0.5 0.644], 'LineWidth', 2)
hold on
plot([2,0],[0,0], '-.','color', [0.95 0.33 0.33],'LineWidth', 1.6)
hold on 
plot([0.68 0.68], [-1.2 0.2],'-.', 'color', [0.95 0.33 0.33],'LineWidth', 1.6)
hold on
plot(0.68,0,'k*','LineWidth',3)
text(0.70,0.05,'$(0.68, 0)$: transition point','interpreter','latex','FontSize',18)
set(gca,'TickLabelInterpreter','latex')
set(gca, 'FontSize', 14)
xlabel('$\mu$: the degree of each node divided by $n$','interpreter','latex','FontSize',20)
ylabel('$n^{-1}\cdot \lambda_2(\nabla^2 E)$','interpreter','latex','FontSize',20)
title('$n^{-1}\cdot \lambda_2(\nabla^2 E)$ vs.\ $\mu$,\quad $n = 1000$','interpreter','latex','FontSize',25,'fontWeight','bold')
print('-depsc','lambda2_mu.eps')






