
% make a sample input:
arraySize = 8;
X = 2*pi*rand(arraySize);

eIn = exp(1i.*X);
eOut = fftshift(fft2(eIn));

CT = cbrewer('seq', 'Greys', 100);

% plot:
figure(1); clf;
colormap(CT)
imagesc(abs(eOut).^2)
colorbar

%% Make a target:
x = linspace(-1,1,arraySize);
[X, Y] = meshgrid(x, x);

w1 = 0.2/3;
w2 = 0.1/3;
r = +0.5/3;
target = exp(-X.^2/(2*w1^2)) + ...
         exp(-( sqrt(X.^2+Y.^2)-r ).^2/(2*w2^2));

figure(2); clf;
colormap(CT)
imagesc(x,x,target)
colorbar
title('Target')


%% Do gradient checking:

costFunction = @(x) seGradient(target, x, arraySize);

% create an initial input:
X = 2*pi*rand(arraySize);
X = X(:);

[C, grad] = costFunction(X);

% numerically evaluate gradient:
numgrad = computeNumericalGradient(costFunction, X);

[grad numgrad]

%% Do CG optimisation:

options = optimset('MaxIter', 50);

costFunction = @(p) backPropGradientClf(X_train, y_train, ...
                                        p, ...
                                        N_in, N_hidden, N_out, ...
                                        lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, nn_params, options);