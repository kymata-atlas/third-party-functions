function Energy = modelEnergy(U,V, fx0,fy0,ft0, sigmaX,sigmaY,sigmaT)
% Model for motion energy, implements Eq. (9) from Heeger (1988).
H2 = (U*fx0 + V*fy0 + ft0).^2;
H3 = (U*sigmaX*sigmaT).^2 + (V*sigmaY*sigmaT).^2 + (sigmaX*sigmaY)^2;
Energy = exp(-4*pi^2*sigmaX^2*sigmaY^2*sigmaT^2 * H2./H3);