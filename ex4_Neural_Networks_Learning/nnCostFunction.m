function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%
% 1. Implement forword propagation to get h_theta(x)
%

% LAYER 1
X = [ones(m,1) X];         % Bias - adding a row of m ones

% LAYER 2 - hidden layer
z2 = X*Theta1';                          % Compute z2 - input of g(sifmoid function) 
a2 = [ones(size(z2, 1), 1) sigmoid(z2)]; % g(z2) add a row of ones 

% LAYER 3 - output layer
z3 = a2*Theta2';
a3 = sigmoid(z3);
% The output layer is hypothesis function
h = a3;

% 
% 2. J function computation
%

% Reguralarization - computing penalty
penalty = sum(sum(Theta1(:, 2:end).^2, 2))+sum(sum(Theta2(:, 2:end).^2, 2));
I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end
J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m + lambda*penalty/(2*m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Y = zeros(num_labels, m);
for i=1:num_labels,
    Y(i,:) = (y==i);
endfor

for t = 1:m,
    %Set the input layer’s values (a(1)) to the t-th training example x(t)
    a1 = X(t,:);
    %Perform a feedforward pass, computing the activations (z(2), a(2), z(3), a(3)) 
    %for layers 2 and 3.
    z2 = Theta1 * a1';
    a2 = sigmoid(z2);
    a2 = [1 ; a2]; % add a+1 term to ensure that the vectors of activations 
                   % for layers a(1) and a(2) also include the bias unit.
    %LAYER 3    
    a3 = sigmoid(Theta2 * a2);
    %For each output unit k in layer 3 set δ(3) = (a(3) − yk)
    d3 = a3 - Y(:,t);
    % Bias
    z2 = [1 ; z2];
    %For the hidden layer l = 2, set δ(2) = 􏰀Θ(2)􏰁T δ(3). ∗ g′(z(2))
    d2 = (Theta2' * d3) .* sigmoidGradient(z2);
    d2 = d2(2:end);
    %Accumulate the gradient
    Theta2_grad = (Theta2_grad + d3 * a2');
    Theta1_grad = (Theta1_grad + d2 * a1);
endfor;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,1) = Theta1_grad(:,1)./m;

Theta2_grad(:,1) = Theta2_grad(:,1)./m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)./m + ( (lambda/m) * Theta1(:,2:end) );

Theta2_grad(:,2:end) = Theta2_grad(:,2:end)./m + ( (lambda/m) * Theta2(:,2:end) );

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
