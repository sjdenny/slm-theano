% SJD 14 September 2014
% Calculate Squared Error cost function and gradient:

function [C, grad] = seGradient(target, x, npixels)

% unroll x into image:
phi = reshape(x, npixels, npixels);

% compute FFT of phi:
psi = exp(1i.*phi);
psiTilde = fftshift(fft2(psi));
% psiTilde = fft2(psi);

err = (target - abs(psiTilde).^2);
C = sum(sum(err.^2)) / (2*npixels^2);

% compute gradient:
% X = ifft2(err.*conj(psiTilde));
X = ifft2(ifftshift(err).*conj(psiTilde));

grad = real(conj(psiTilde).*X);

% roll up for return
grad = grad(:);

end