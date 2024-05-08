clc;
close all;
clear;
% PSD data logger inspection

%22
file_name = "2024_5_4_19_25.csv";
data = readtable(file_name,'VariableNamingRule', 'preserve');
% cdata = csvread(file_name, 1);
% psd1_vx1 = data{:,2};
% psd1_vx2 = data{:,3};
% psd1_vy1 = data{:,4};
% psd1_vy2 = data{:,5};
% 
% psd1_dx = (psd1_vx2 + psd1_vy1) - (psd1_vx1 + psd1_vy2);
% psd1_dy = (psd1_vx2 + psd1_vy2) - (psd1_vx1 + psd1_vy1);
% sigma =  (psd1_vx1 + psd1_vx2 + psd1_vy1 + psd1_vy2);
% 
% x = psd1_dx./sigma * 2.25;
% y = psd1_dy./sigma * 2.25;
% 
% figure(1);
% plot(x,y);
% grid on;
% xlim([-2.25,2.25]);
% ylim([-2.25,2.25]);
% 
% psd1_vx1 = data{:,11};
% psd1_vx2 = data{:,12};
% psd1_vy1 = data{:,13};
% psd1_vy2 = data{:,14};

% psd1_dx = (psd1_vx2 + psd1_vy1) - (psd1_vx1 + psd1_vy2);
% psd1_dy = (psd1_vx2 + psd1_vy2) - (psd1_vx1 + psd1_vy1);
% sigma =  (psd1_vx1 + psd1_vx2 + psd1_vy1 + psd1_vy2);
% 
% x = psd1_dx./sigma * 2.25;
% y = psd1_dy./sigma * 2.25;
% 
% figure(1);
% plot(x,y);
% grid on;
% xlim([-2.25,2.25]);
% ylim([-2.25,2.25]);


time = data{:,1};
x1 = data{:,6};
y1 = data{:,7};
sigma1 = data{:,8};
x2 = data{:, 15};
y2 = data{:, 16};
sigma2 = data{:, 17};



figure(2);
subplot(3,1,1);
plot(time, x1);
grid on;
title('PSD 1');
xlabel('time(s)');
ylabel('X(mm)');
ylim([-2.25,2.25]);

subplot(3,1,2);
plot(time,y1);
grid on;
xlabel('time(s)');
ylabel('Y(mm)');
ylim([-2.25,2.25]);


subplot(3,1,3);
plot(time,sigma1);
grid on;
xlabel('time{s}');
ylabel('sigma(V)');

figure(3);

subplot(3,1,1);
plot(time, x2);
grid on;
title('PSD 2');
xlabel('time(s)');
ylabel('X(mm)');
ylim([-2.25,2.25]);

subplot(3,1,2);
plot(time,y2);
grid on;
xlabel('time(s)');
ylabel('Y(mm)');
ylim([-2.25,2.25]);

subplot(3,1,3);
plot(time, sigma2);
grid on;
xlabel('time{s}');
ylabel('sigma(V)');

figure(4);
plot(x1,y1);
grid on;
title('PSD 1');
xlabel('X(mm)');
ylabel('Y(mm)')
xlim([-2.25,2.25]);
ylim([-2.25,2.25]);

figure(5);
plot(x2,y2);
grid on;
title('PSD 2')
xlabel('X(mm)');
ylabel('Y(mm)')
xlim([-2.25,2.25]);
ylim([-2.25,2.25]);
