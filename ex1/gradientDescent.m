function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y);                   % number of training examples
J_history = zeros(num_iters, 1);
X_2_col_vec = X(:,2);            %extract the second

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    hypothesis_vec = theta(1) + theta(2) * X_2_col_vec;
    theta_0 = theta(1) - alpha * (1/m) * sum(hypothesis_vec - y);
    theta_1 = theta(2) - alpha * (1/m) * sum((hypothesis_vec - y).*X_2_col_vec);

    theta = [theta_0; theta_1];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
%J(theta) = transpose(theta)*X