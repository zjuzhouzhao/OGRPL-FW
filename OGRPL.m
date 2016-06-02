function [WK] = OGRPL(R,P,X,L,lambda,K) 
%% The OGRPL-FW algorithm, 
%  At k-step, solving a nuclear norm constrained problem  
%     min FK = \sum{i=1}{k} ||(P==i).*(R-W{i}'*X)||_F^2 + lambda*tr(X'WLW'X)  s.t. ||W{i}||_* <= r 
% R : the rating matrix, 
% P : rating indices, while P(i,j) = k means rating R(i,j) is at the k-th round.
% X : the feature content matrix of items, 
% L : the laplacian matrix
% W{i} : the target preference matrix at i-step,
% lambda : the regularization term.
% K : timestamp
% d : dimension of the feature content vector of items.

%% Initialize
    W = cell(K+1,1);
    d = size(X,1);
    [n, m] = size(R);
    W{1} = rand(d,n);
    grad = cell(K,1); % grad{k} is the gradient of FK
    inner_iter = 1; % In each step, a inner iteration conduct the gradient descent for FK
    alpha = 0.0009; % learning rate, perform linear search.
    
%% Optimization
    for tau = 1 : K
        for i = 1 : inner_iter
            if (tau == 1)
                grad{tau} = ((P == tau).*(W{tau}'*X - R)*X')';
            else
                grad{tau} = grad{tau-1} + ((P == tau).*(W{tau}'*X - R)*X')';
            end
            var = grad{tau} + 2*lambda*(X*X')*W{tau}*L;
            [A,B,C] = svds(var,1); % return the largest singular value and its corresponding unitary matrix 
            V = A*B*C';
            W{tau} = W{tau} - alpha*V;
        end
        W{tau+1} = W{tau};
        RMSE = norm((P>0).*(R - W{tau}'*X),'fro')/sqrt(sum(sum(P>0)));
        fprintf('k = %d, trainning root mean square error = %.4f\n',tau,RMSE); %update the train error at each step
    end
    
%% return
    WK = W{K+1}; %return the final result.
end

