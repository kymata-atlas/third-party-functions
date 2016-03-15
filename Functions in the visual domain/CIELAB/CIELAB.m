
% This function is an implementation of the CIELAB (also known as the CIEL*A*B*) color appearance model
% -------
% Authors: International Commission on Illumination (CIE) (2008)
% DOI: Joint ISO/CIE Standard, ISO 11664-4:2008(E)/CIE S 014-4/E:2007
% Address: http://cie.co.at/index.php?i_ca_id=485
% Input stream: The average X Y Z tristimulus values over the entire field of vision; the white reference X Y Z tristimulus values (per single frame window).

% For historical background to this model, see Chapter 10 of Color Appearance Models, Third Edition. Mark D. Fairchild. © 2013 John Wiley & Sons, Ltd. Published 2013 by John Wiley & Sons, Ltd
 

function [ L, A, B ] = CIELAB(X, Y, Z, WhiteRefX, WhiteRefY, WhiteRefZ)
   
    xr=X/WhiteRefX;
    yr=Y/WhiteRefY;
    zr=Z/WhiteRefZ;

    % eq. 1 (equation number refers to ISO specification)

    L = 116 * nonLinearCompression(yr) - 16;

    % eq. 2

    A = 500 * (nonLinearCompression(xr) - nonLinearCompression(yr));

    % eq. 3

    B = 200 * (nonLinearCompression(yr) - nonLinearCompression(zr));

end


function compressed_color_value = nonLinearCompression(color_value)

    % CIELAB nonlinear compression (eqs. 4-9)
    % ------
    % Models the compressive response typically found between physical energy measurements and perceptual responses of colors (Stevens 1961)).

    if (value > 0.008856)
        compressed_color_value = color_value^(1/3);
    else
        compressed_color_value = 7.787*color_value + 16/116;
    end

end



