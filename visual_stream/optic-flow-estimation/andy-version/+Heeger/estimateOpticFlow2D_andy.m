function [Dx, Dy, L] = estimateOpticFlow2D(ImgSeq, opt)
% estimateOpticFlow2D
%   ImgSeq  - Image sequence as a cube with dimensions: 
%             height x width x frames.
%   opt     - Structure with options:
%             * fxy     - Spatial frequency in cycles per pixel.
%             * oNum    - Number of orientations in space.
%             * sigmaX  - Gaussian standard deviation of Gabor in x.
%             * sigmaY  - Gaussian standard deviation of Gabor in y.
%             * TempFrq - Temporal frequency vector in cycles per frame. 
%             * sigmaT  - Gaussian standard deviation of Gabor in t.
%             * sigmaV  - Standard deviation for Gaussian distribution
%                         function that is used to compute likelihoods.
%             * VelVecX - X-components of velocity vector.
%             * VelVecY - Y-components of velocity vector.
%
% RETURN
%   Dx      - X-component of optic flow.
%   Dy      - Y-component of optic flow.
%             Computes the optic flow for the center frame of the sequence,
%             thus, Dx and Dy have dimensions: height x width.
%   L       - Likelihood values for sampled velocity space. 
%             Dimensions: height x width x vyNum x vxNum.
%
% DESCRIPTION
%   An implementation of the idea of 
%   Heeger, D.J. (1988). Optical flow using spatiotemporal filters. 
%       International Journal of Computer Vision 1(4), 279-302.
%   This implementation does not realize multiple levels using an image
%   pyramid.
%
%   Copyright (C) 2013  Florian Raudies, 01/02/2013, Boston University.
%   License, GNU GPL, free software, without any warranty.

% Set default values for parameters of the method.

clear
first_frame = 6;

ImgSeq        = double(stream_our_stim(first_frame, 15, 'greyscale'));
% Set default values for parameters of the method.
opt         = struct();      % UNITS
opt.fxy     = 1/10;           % cycles per pixel
opt.oNum    = 4;             % -
opt.sigmaX  = 4;             % pixels
opt.sigmaY  = 4;             % pixels
opt.TempFrq = [-1/4 0 1/4];  % cycles per frame
opt.sigmaT  = 1;             % frames
opt.sigmaV  = 5*10^-1;         % -
opt.VelVecX = linspace(-1,1,15); % pixels per frame
opt.VelVecY = linspace(-1,1,15); % pixels per frame
% Retrieve parameter values.
fxy     = opt.fxy;
oNum    = opt.oNum;
TempFrq = opt.TempFrq;
sigmaX  = opt.sigmaX;
sigmaY  = opt.sigmaY;
sigmaT  = opt.sigmaT;
sigmaV  = opt.sigmaV;
VelVecX = opt.VelVecX;
VelVecY = opt.VelVecY;
SpaceOri= (0:oNum-1)*pi/oNum;
[Vy, Vx] = ndgrid(VelVecY,VelVecX);
yNum    = size(ImgSeq,1);
xNum    = size(ImgSeq,2);
ftNum   = length(TempFrq);
vyNum   = size(Vy,1);
vxNum   = size(Vy,2);
% *************************************************************************
% Generate Gabor filter kernels for separable filtering. 
%   See Heeger, D. (1987). Model for the extraction of image flow. 
%       Journal of Optical Society, Series A, 4(8), 1455-1471.
% *************************************************************************
fKernel     = @(sigma)      (-3*sigma:+1:+3*sigma);
fGaussian   = @(Z,sigma)    1/(sqrt(2*pi)*sigma) * exp(-Z.^2/(2*sigma^2));
fGaborSin   = @(Z,f,sigma)  fGaussian(Z,sigma) .* sin(2*pi*f*Z);
fGaborCos   = @(Z,f,sigma)  fGaussian(Z,sigma) .* cos(2*pi*f*Z);
KernelX     = fKernel(sigmaX);
KernelY     = fKernel(sigmaY);
KernelT     = fKernel(sigmaT);
kxNum       = length(KernelX);
kyNum       = length(KernelY);
ktNum       = length(KernelT);
Binomi8     = [1 8 28 56 70 56 28 8 1]/256;
WindowY     = reshape(Binomi8, [9 1 1]); % Parseval window
WindowX     = reshape(Binomi8, [1 9 1]);
WindowT     = reshape(Binomi8, [1 1 9]);
wtNum       = length(Binomi8);
% Check if the provided sequence contains enough frames.
tNum    = size(ImgSeq,3);
if tNum<(ktNum+wtNum-1), 
    error('MATLAB:frameErr', ['This method requires at least %d frames ',...
        'but only %d frames were provided!'], ktNum+wtNum-1, tNum);
end
% Select the center number of ktNum+wtNum frames.
center  = ceil(tNum/2);
SelT    = center + ( -(ktNum-1)/2-(wtNum-1)/2 : +(ktNum-1)/2+(wtNum-1)/2 );
ImgSeq  = ImgSeq(:,:,SelT);
% *************************************************************************
% Laplacian spatial filtering to remove DC component from image sequence.
% *************************************************************************
%Laplace     = [1 -2 1];
%LaplaceY    = reshape(Laplace, [3 1 1]);
%LaplaceX    = reshape(Laplace, [1 3 1]);
%ImgSeq      = imfilter(imfilter(ImgSeq, LaplaceX, 'same', 'replicate'), ...
%                                        LaplaceY, 'same', 'replicate');
% *************************************************************************
% Compute the motion energy for spatial and temporal frequencies.
% *************************************************************************
L = zeros(yNum, xNum, vyNum, vxNum);  % likelihood values
for iSpaceFreq = 1:oNum,
    disp(['iSpaceFreq: ', num2str(iSpaceFreq)]);
    ModelEnergy = zeros(vyNum, vxNum, ftNum);
    DataEnergy  = zeros(yNum,  xNum,  ftNum);
    % Define spatial frequencies.
    fx0 = cos(SpaceOri(iSpaceFreq))*fxy;
    fy0 = sin(SpaceOri(iSpaceFreq))*fxy;
    % Define spatial part of Gabor filters.
    Gabor.SinX = reshape(fGaborSin(KernelX,fx0,sigmaX), [1 kxNum 1]);
    Gabor.CosX = reshape(fGaborCos(KernelX,fx0,sigmaX), [1 kxNum 1]);
    Gabor.SinY = reshape(fGaborSin(KernelY,fy0,sigmaY), [kyNum 1 1]);
    Gabor.CosY = reshape(fGaborCos(KernelY,fy0,sigmaY), [kyNum 1 1]);
    for iTempFreq = 1:ftNum,
        disp(['iTempFreq: ', num2str(iTempFreq)]);
        ft0 = TempFrq(iTempFreq);
        % Define temporal part of Gabor filters.
        Gabor.SinT = reshape(fGaborSin(KernelT,ft0,sigmaT), [1 1 ktNum]);
        Gabor.CosT = reshape(fGaborCos(KernelT,ft0,sigmaT), [1 1 ktNum]);
        % Compute Gabor filter output and power.
        FilterEnergy = abs(filterSepGabor(ImgSeq, Gabor)).^2;
        % Approximate Parseval's theorem by filtering with binomial kernel.
        DataEnergy(:,:,iTempFreq) = 1/(8*pi^3) ...
                                *imfilter(imfilter(convn(FilterEnergy, ...
                                    WindowT, 'valid'), ...
                                    WindowX, 'same', 'replicate'), ...
                                    WindowY, 'same', 'replicate');
        % Compute the model energy.
        ModelEnergy(:,:,iTempFreq) = modelEnergy(Vx,Vy,fx0,fy0,ft0,...
                                                 sigmaX,sigmaY,sigmaT);
    end
    % Compute normalization coefficients for all temporal frequencies of
    % one orientation.
    Rbar = repmat(squeeze(mean(DataEnergy,3)),[1 1 vyNum vxNum]);
    Mbar = shiftdim(repmat(squeeze(mean(ModelEnergy,3)),[1 1 yNum xNum]),2);
    MbarDivRbar = Mbar./(Rbar+eps);
    for iTempFreq = 1:ftNum,
        M = shiftdim(repmat(ModelEnergy(:,:,iTempFreq),[1 1 yNum xNum]),2);
        R = repmat(DataEnergy(:,:,iTempFreq),[1 1 vyNum vxNum]);
        L = L + (M-MbarDivRbar.*R).^2;
    end
end
% Parallel computation of motion velocities by sampling and max selection.
L   = reshape(exp(-1/(2*sigmaV^2)*L),[yNum xNum vyNum*vxNum]);
[~, Index] = max(L,[],3);
Vx  = reshape(Vx,[vyNum*vxNum 1]);
Vy  = reshape(Vy,[vyNum*vxNum 1]);
Dx  = Vx(Index);
Dy  = Vy(Index);

%Another read-out computes the likelihood weighted vector sum.
%Lsum = sum(L,3) + eps;
%Dx = sum(L.*shiftdim(repmat(Vx(:),[1 yNum xNum]),1),3)./Lsum;
%Dy = sum(L.*shiftdim(repmat(Vy(:),[1 yNum xNum]),1),3)./Lsum;
L   = reshape(L,[yNum xNum vyNum vxNum]);


imagesc(Dx);
title(sprintf(['Heeger, default settings, frame ', num2str(first_frame)]))
map = [1 0 0
       0.75 0 0
       0.5 0 0
       0.25 0 0
       0 0 0
       0 0.25 0
       0 0.5 0
       0 0.75 0
       0 1 0 ];
colormap(map)
caxis([-1 1])

