function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

m = size(y);
% for i=1:m
% 	if y(i)==0
% 	plot(X(i, 1), X(i, 2), 'k+');
% 	hold on;
%     elseif y(i)==1
% 	plot(X(i, 1), X(i, 2), 'ko');
% 	hold on;
% 	end 
% end

idx0 = find(~y);
idx1 = find(y);
plot(X(idx0, 1), X(idx0, 2), 'k+', X(idx1, 1), X(idx1, 2), 'ko');

hold on;

% =========================================================================



hold off;

end
