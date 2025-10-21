clear all; close all; clc 

%% Assignment 2: Problem 1

%% Part 0: generate data 
rng(42); 
% Specify number of feature dimensions 
NTrain = [50; 500; 5000]; % Specify number of training samples for experiments
NVal = 10000; % Specify number of validation samples for experiments

% Class priors
P0 = 0.6;
P1 = 0.4;
priors = [P0 P1];
% Mixture weights
w01 = 0.5; w02 = 0.5; w11 = 0.5; w12 = 0.5;

% Means
m01 = [-0.9; -1.1];
m02 = [0.8; 0.75];
m11 = [-1.1; 0.9];
m12 = [0.9; -0.75];
% Covariance matrices (same for all components)
C = [0.75 0; 0 1.25];
epsilon = 1e-3; % stopping criterion threshold/tolerance
alpha = 0.01; % step size for gradient descent methods

% Generate datasets and visualize
D_train_5000 = generate_dataset(5000, P0, P1, w01, w02, w11, w12, m01, m02, m11, m12, C);

figure(1);
gscatter(D_train_5000.X(1,:), D_train_5000.X(2,:), D_train_5000.L, 'rb', 'ox', 8);
hold on
scatter(-0.9, -1.1, "*y")
hold on
scatter(0.8, 0.75, "*y")
hold on
scatter(-1.1, 0.9, "*y")
hold on
scatter(0.9, -0.75, "*y")
xlabel('x_1');
ylabel('x_2');
title('Generated Samples by True Class Label');
legend({'Class 0','Class 1'}, 'Location', 'best');

%% Part 1: Find minimum probability of error and plot ROC curve 
% Generate iid validation samples
disp('Generating the validation data set.'),
data = generate_dataset(NVal, P0, P1, w01, w02, w11, w12, m01, m02, m11, m12, C);
componentLabels = data.L;
xVal = data.X;
n = size(xVal,1);
labelsVal = componentLabels; % convert 0/1 component labels to 0/1 class labels
NcVal = [length(find(labelsVal==0)), length(find(labelsVal==1))];

X = data.X;
true_labels = data.L;
N = size(X, 2);

% Preallocate
px_L0 = zeros(1, N);
px_L1 = zeros(1, N);

% Evaluate class-conditional pdfs
for i = 1:N
    x = X(:, i)';
    % p(x | L = 0)
    p0 = w01 * mvnpdf(x, m01', C) + w02 * mvnpdf(x, m02', C);
    % p(x | L = 1)
    p1 = w11 * mvnpdf(x, m11', C) + w12 * mvnpdf(x, m12', C);
    % Store
    px_L0(i) = p0;
    px_L1(i) = p1;
end

% Compute posterior scores (likelihood ratio)
%scores = (P_L0 * px_L0) ./ (P_L1 * px_L1);
numerator = P1 * px_L1;
denominator = P0 * px_L0 + P1 * px_L1;
scores = numerator ./ denominator;

% Bayes classifier decision at threshold t=1
predicted_labels = scores > 1;

% Estimate min-P(error)
num_errors = sum(predicted_labels ~= true_labels);
min_P_error = num_errors / N;
fprintf('Estimated minimum probability of error: %.4f\n', min_P_error);

% Compute ROC curve
thresholds = logspace(-2, 2, 200); % log scale
TPR = zeros(size(thresholds));
FPR = zeros(size(thresholds));
Perror = zeros(size(thresholds));


for j = 1:length(thresholds)
    t = thresholds(j);
    preds = scores > t;

    TP = sum((preds == 1) & (true_labels == 1));
    FP = sum((preds == 1) & (true_labels == 0));
    FN = sum((preds == 0) & (true_labels == 1));
    TN = sum((preds == 0) & (true_labels == 0));
    Perror(j) = sum(preds~=true_labels)/length(true_labels); % Errors (decide 1 when L=0 or decide 0 when L=1)

    TPR(j) = TP / (TP + FN); % Sensitivity
    FPR(j) = FP / (FP + TN); % 1 - Specificity
end


[Pe_min, idxMin] = min(Perror);
optimized_tau = thresholds(idxMin);
Ptp_emp = TPR(idxMin);
Pfp_emp = FPR(idxMin);

% Mark optimal point (threshold = 1)
bayes_idx = find(thresholds <= 0.5, 1, 'last'); % Since we're sweeping from high to low
plot(FPR(bayes_idx), TPR(bayes_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
legend('ROC Curve', 'Bayes Optimal Point', 'Location', 'southeast');

% Plot ROC curve 
figure(2)
plot(FPR, TPR,'b.-','LineWidth',1.5),
hold on
plot(Pfp_emp, Ptp_emp, 'ro','MarkerSize',10,'LineWidth',2); % mark empirical min error
xlabel('PFP'),ylabel('PTP'), title('ROC Curve for ERM Discriminant Scores')

%% Part 2: Train 3 separate logistic-linear and logistic-quadratic function based approximations and estimate error
for i=1:length(NTrain) % Conduct experiments with different amounts of training samples
    disp(strcat('Generating the training data set; Ntrain = ',num2str(NTrain(i)))),
    % Generate iid training samples as specified
    data = generate_dataset(NTrain(i), P0, P1, w01, w02, w11, w12, m01, m02, m11, m12, C);
    xTrain{i} = data.X;
    componentLabels = data.L;
    labelsTrain{i} = componentLabels; % convert 0/1 component labels to 0/1 class labels
    NcTrain{i} = [length(find(labelsTrain{i}==0)), length(find(labelsTrain{i}==1))];
    
    disp('Training the logistic-linear model with gradient descent.'),
    % Deterministic (batch) gradient descent 
    % Uses all samples in training set for each gradient calculation
    paramsGD.type = 'batch';
    paramsGD.ModelType = 'logisticLinear';
    paramsGD.stepSize = alpha;
    paramsGD.stoppingCriterionThreshold = epsilon;
    paramsGD.minIterCount = 10;
    [wGradDescentLin,zLin] = gradientDescent_binaryCrossEntropy(xTrain{i},labelsTrain{i},paramsGD);
 
    % Shared initial weights for iterative gradient descent methods...
    disp('Training the logistic-quadratic model with gradient descent.'),
    % Deterministic (batch) gradient descent 
    % Uses all samples in training set for each gradient calculation
    paramsGD.type = 'batch';
    paramsGD.ModelType = 'logisticQuadratic';
    paramsGD.stepSize = alpha;
    paramsGD.stoppingCriterionThreshold = epsilon;
    paramsGD.minIterCount = 10;
    [wGradDescentQuad,zQuad] =  gradientDescent_binaryCrossEntropy(xTrain{i},labelsTrain{i},paramsGD);
    
  
    %Linear: use validation data(10k points) and make decisions
    test_set_L=[ones(1,NVal); xVal];
    decision_L_GD=wGradDescentLin'*test_set_L>=0; 

    %linear: plot all decision and boundary line
    figure(5); 
    error_L_GD(i)=plot_classified_data(decision_L_GD,labelsVal, NcVal, priors ,... 
        [1,3,i], test_set_L,wGradDescentLin,'L',n);
    title(['2D Linear GD Classification Based on ' , num2str(NTrain(i)),' Samples']);

    %Quadratic: use validation data (10k points) and make decisions
    test_set_Q=[ones(1,NVal) ;xVal ]; 
    for r = 1:n
        for c = 1:n
            test_set_Q = [test_set_Q;xVal(r,:).*xVal(c,:)];
        end
    end
    decision_Q_GD=wGradDescentQuad'*test_set_Q>=0; 
    
    %Quadratic: plot all decisions and boundary contour 
    figure(9)
    error_quad_GD(i)= plot_classified_data(decision_Q_GD, labelsVal,NcVal,...
       priors,[1,3,i],test_set_Q,wGradDescentQuad,'Q',n);
    title(['2D Quadratic GD Classification Based on ', num2str(NTrain(i)),' Samples']);

end 


%% Print all calculated error values 

fprintf('<strong>Logistic Regeression Total Error Values</strong>\n');
fprintf('Training Set Size \tLinear Approximation Error (%%)\tQuadratic Approximation Error (%%)\n');
fprintf('\t  %i\t\t\t\t\t %.2f%%\t\t\t\t\t\t\t%.2f%%\n',[NTrain; error_L_GD';error_quad_GD']); 

% Helper function for generating data 
function D = generate_dataset(N, P0, P1, w01, w02, w11, w12, m01, m02, m11, m12, C)
    D = struct('X', [], 'L', []);
    D.X = zeros(2, N);
    D.L = zeros(1, N);

    for i = 1:N
        % Sample class label L
        L = rand() < P0; % L=1 if class 0, L=0 if class 1 (since true when rand < 0.6)

        if L == 1 % Class 0
            % Choose one of the two Gaussians in class 0
            if rand() < w01
                x = mvnrnd(m01, C)';
            else
                x = mvnrnd(m02, C)';
            end
            label = 0;
        else % Class 1
            if rand() < w11
                x = mvnrnd(m11, C)';
            else
                x = mvnrnd(m12, C)';
            end
            label = 1;
        end

        D.X(:, i) = x;
        D.L(i) = label;
    end
end
