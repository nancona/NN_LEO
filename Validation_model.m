% ================================
%   CODE USED TO PLOT ---> PAPER
% ================================

clc, clear all
% sample interval
starting_sample = 7000;
nsample_rmse = 999;
nsample_plot = 99;

file_1 = './results_model_nn/20k_100/prediction.txt';
file_2 = './results_model_nn/20k_100/test_dataset.txt';

fileID_pred = fopen(file_1,'r');
fileID_vald = fopen(file_2,'r');
prediction = importdata(file_1,'\t');
validation = importdata(file_2,'\t');

i = starting_sample;

for i = 1:length(prediction(1,:))
    RMSE(i) = [sqrt(immse(prediction(:,i), validation(:,i)))];
end
%%

% ====================
%   PLOT SETTINGS RB
% ====================

MarkerSize = 9;
LineWidth = 1;
fs = 14;

Ts = 0.03;
ylabel_plot = {'$\alpha$ [rad]', '$\dot\alpha$ [rad*$s^{-1}$]'};
plot_title = {'Torso X-Position', 'Torso Z-Position', 'Torso Angle', ...
              'Left Hip Angle', 'Right Hip Angle', 'Left Knee Angle', ...
              'Right Knee Angle', 'Left Ankle Angle', 'Right Ankle Angle', ...
              'Torso X-Velocity', 'Torso Z-Velocity', 'Torso Velocity', ...
              'Left Hip Velocity', 'Right Hip Velocity', 'Left Knee Velocity', ... 
              'Right Knee Velocity', 'Left Ankle Velocity', 'Right Ankle Velocity'
              };
set(0,'DefaultAxesFontName','times');
set(0,'DefaultTextFontName','times');
set(0,'DefaultAxesFontSize',fs);
set(0,'DefaultTextFontSize',fs);
for i = 1:length(prediction(1,:))
    f1 = figure(i); hold on;
    f1.Renderer = 'Painters';
    p2 = plot((1:nsample_plot) * Ts, prediction(1:nsample_plot,i), 'b', 'MarkerSize', MarkerSize);
    p3 = plot((1:nsample_plot) * Ts, validation(1:nsample_plot,i), 'r', 'MarkerSize', MarkerSize);
    xlabel('$t$ [s]', 'FontSize', fs, 'Interpreter', 'latex');
    if i<9    
        ylabel(ylabel_plot(1), 'FontSize', fs, 'Interpreter', 'latex');
    else
        ylabel(ylabel_plot(2), 'FontSize', fs, 'Interpreter', 'latex');
    end
    title(plot_title(i), 'FontSize', fs+1);
    ax = gca; ax.FontSize = fs;
    grid on;
    % axis([0 size(ytrial_5, 1)*Ts -0.22*pi 0.33*pi]);
    legend({'Predicted', 'Validation'}, 'FontSize', fs-2, 'Interpreter', 'latex', 'Location','southeast');
    
%     h = gcf;
%     set(h,'Units','Inches');
%     pos = get(h,'Position');
%     set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
%     print(h,plot_time(i),'-dpdf')
end
%% save to pdf

h = gcf;
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
print(h,'Right Ankle Angle','-dpdf')
% 
% plot(ytrial_trs_alpha_6);
% hold on;
% plot(outputv);
% title('Torso Angle');
% legend('Predicted', 'Validation')
