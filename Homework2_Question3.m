%% Homework 2, Question 3
clc; clear all; close all;

% Parameters
sigma_r = 0.3;         % Range measurement noise standard deviation
sigma_x = 0.25;        % Prior std dev in x
sigma_y = 0.25;        % Prior std dev in y
sigma_x2 = sigma_x^2;
sigma_y2 = sigma_y^2;
sigma_r2 = sigma_r^2;

% Grid for evaluating objective function
x_vals = linspace(-2, 2, 100);
y_vals = linspace(-2, 2, 100);
[X, Y] = meshgrid(x_vals, y_vals);

% To store global min/max values of the objective across all K (for consistent contours)
all_J_min = inf;
all_J_max = -inf;
J_map_all = {};

figure;
for K = 1:4
    % --- Generate true position in unit circle ---
    theta = 2*pi*rand;
    r = sqrt(rand);  % Uniform in circle
    x_true = r * cos(theta);
    y_true = r * sin(theta);
    true_pos = [x_true; y_true];

    % --- Landmarks ---
    angles = linspace(0, 2*pi, K+1); angles(end) = [];
    landmarks = [cos(angles); sin(angles)];

    % --- Range measurements ---
    ranges = zeros(1, K);
    for i = 1:K
        d_i = norm(true_pos - landmarks(:, i));
        noisy_range = -1;
        while noisy_range < 0
            noisy_range = d_i + sigma_r * randn;
        end
        ranges(i) = noisy_range;
    end

    % --- Evaluate objective over grid ---
    J = zeros(size(X));
    for i = 1:numel(X)
        x = X(i);
        y = Y(i);
        pos = [x; y];

        range_cost = 0;
        for j = 1:K
            d = norm(pos - landmarks(:, j));
            range_cost = range_cost + (ranges(j) - d)^2 / sigma_r2;
        end

        prior_cost = x^2 / sigma_x2 + y^2 / sigma_y2;
        J(i) = range_cost + prior_cost;
    end

    % Store for common contour level plotting later
    J_map_all{K} = J;
    all_J_min = min(all_J_min, min(J(:)));
    all_J_max = max(all_J_max, max(J(:)));

    % --- Plotting ---
    subplot(2,2,K);
    contourf(X, Y, J, 20, 'LineColor', 'none'); hold on;
    plot(x_true, y_true, 'r+', 'MarkerSize', 10, 'LineWidth', 2);
    plot(landmarks(1,:), landmarks(2,:), 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);
    title(['K = ' num2str(K)]);
    axis equal; xlim([-2 2]); ylim([-2 2]);
    xlabel('x'); ylabel('y');
    colorbar;
end

sgtitle('MAP Objective Function Contours for K = 1 to 4');