clc;
close all;
clear;
% PSD data logger inspection

%22
file_name = "good.csv";
data = readtable(file_name,'VariableNamingRule', 'preserve');

x_ideal = data{:, 20};
y_ideal = data{:, 21};
z_ideal = data{:, 22};





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
scatter(x1,y1);
grid on;
title('PSD 1');
xlabel('X(mm)');
ylabel('Y(mm)')
xlim([-2.25,2.25]);
ylim([-2.25,2.25]);

figure(5);
scatter(x2,y2);
grid on;
title('PSD 2')
xlabel('X(mm)');
ylabel('Y(mm)')
xlim([-2.25,2.25]);
ylim([-2.25,2.25]);


figure(6);
scatter3(x_ideal, y_ideal, z_ideal, '*');
grid on;
xlabel('X(mm)');
ylabel('Y(mm)');
zlabel('Z(mm');
