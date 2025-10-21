clear all; close all; clc

%% Assignment 2, Question 2
Ntrain = 100; 
data = generateData(Ntrain);
figure(1), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Training Dataset'),
xTrain = data(1:2,:); yTrain = data(3,:);

Nvalidate = 1000; 
data = generateData(Nvalidate);
figure(2), plot3(data(1,:),data(2,:),data(3,:),'.'), axis equal,
xlabel('x1'),ylabel('x2'), zlabel('y'), title('Validation Dataset'),
xValidate = data(1:2,:); yValidate = data(3,:);


% Construct design matrix Phi for cubic polynomial model
Phi_train = getCubicFeatures(xTrain);    % Size: [10 x Ntrain]
Phi_validate = getCubicFeatures(xValidate);  % Size: [10 x Nvalidate]

%% Maximum Likelihood Estimator (ML)
w_ml = (Phi_train * Phi_train') \ (Phi_train * yTrain');

% Predict on validation set
y_pred_ml = w_ml' * Phi_validate;
ml_mse = mean((y_pred_ml - yValidate).^2);

fprintf('ML Average Squared Error: %.4f\n', ml_mse);

%% MAP Estimation for various gamma values
gammas = logspace(-7, 7, 20);  % e.g., from 10^-1 to 10^1
mse_map = zeros(size(gammas));

for i = 1:length(gammas)
    gamma = gammas(i);
    lambda = 1 / gamma;  % Since prior: w ~ N(0, γI), λ = 1/γ
    w_map = (Phi_train * Phi_train' + lambda * eye(size(Phi_train,1))) \ (Phi_train * yTrain');
    y_pred_map = w_map' * Phi_validate;
    mse_map(i) = mean((y_pred_map - yValidate).^2);
end

% Plot Results
figure;
semilogx(gammas, mse_map, 'b-o', 'LineWidth', 1.5);
hold on;
yline(ml_mse, 'r--', 'LineWidth', 2);
xlabel('\gamma (prior variance)');
ylabel('Average Squared Error');
legend('MAP validation error','ML validation error', "Location", "southeast");
title('Validation Error vs. \gamma for MAP and ML Estimators');


%%Find best gamma and corresponding MAP weights
[~, best_idx] = min(mse_map);
best_gamma = gammas(best_idx);
best_lambda = 1 / best_gamma;

% Recompute MAP weights with best gamma
w_best_map = (Phi_train * Phi_train' + best_lambda * eye(size(Phi_train,1))) \ (Phi_train * yTrain');

% Predict on validation set
y_pred_best = w_best_map' * Phi_validate;

% Generate grid for smoother model surface
[x1Grid, x2Grid] = meshgrid(linspace(min(xValidate(1,:)), max(xValidate(1,:)), 50), ...
                            linspace(min(xValidate(2,:)), max(xValidate(2,:)), 50));
xGrid = [x1Grid(:)'; x2Grid(:)'];

% Get features and predict
Phi_grid = getCubicFeatures(xGrid);
yGrid_pred = w_best_map' * Phi_grid;
yGrid_pred = reshape(yGrid_pred, size(x1Grid));


%%
function x = generateData(N)
gmmParameters.priors = [.3,.4,.3]; % priors should be a row vector
gmmParameters.meanVectors = [-10 0 10;0 0 0;10 0 -10];
gmmParameters.covMatrices(:,:,1) = [1 0 -3;0 1 0;-3 0 15];
gmmParameters.covMatrices(:,:,2) = [8 0 0;0 .5 0;0 0 .5];
gmmParameters.covMatrices(:,:,3) = [1 0 -3;0 1 0;-3 0 15];
[x,labels] = generateDataFromGMM(N,gmmParameters);
end 
%%

function [x,labels] = generateDataFromGMM(N,gmmParameters)
% Generates N vector samples from the specified mixture of Gaussians
% Returns samples and their component labels
% Data dimensionality is determined by the size of mu/Sigma parameters
priors = gmmParameters.priors; % priors should be a row vector
meanVectors = gmmParameters.meanVectors;
covMatrices = gmmParameters.covMatrices;
n = size(gmmParameters.meanVectors,1); % Data dimensionality
C = length(priors); % Number of components
x = zeros(n,N); labels = zeros(1,N); 
% Decide randomly which samples will come from each component
u = rand(1,N); thresholds = [cumsum(priors),1];
for l = 1:C
    indl = find(u <= thresholds(l)); Nl = length(indl);
    labels(1,indl) = l*ones(1,Nl);
    u(1,indl) = 1.1*ones(1,Nl); % these samples should not be used again
    x(:,indl) = mvnrnd(meanVectors(:,l),covMatrices(:,:,l),Nl)';
end
end 

%% Function to generate cubic polynomial features
function Phi = getCubicFeatures(x)
% x: 2 x N
x1 = x(1, :);
x2 = x(2, :);

Phi = [
    x1.^3;
    x1.^2 .* x2;
    x1 .* x2.^2;
    x2.^3;
    x1.^2;
    x1 .* x2;
    x2.^2;
    x1;
    x2;
    ones(1, size(x, 2));
];  % Result: 10 x N
end


