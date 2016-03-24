function [a, b, h, A, J, Q ] = CIECAM02(xyz, xyzw, la, yb, para)

    % This function is an implementation of the CIECAM02 color appearance model
    % -------
    % Authors: Nathan Moroney, Mark D. Fairchild, Robert W.G. Hunt, Changjun Li, M. Ronnier Luo and Todd Newman (2002)
    % doi: http://scholarworks.rit.edu/other/143

    % Input stream: The average X Y Z tristimulus values over the entire field of vision; the white reference X Y Z tristimulus values, and various 'viewing parameters' (per single frame window).

    % For some historical background to this model, see Chapter 16 of Color Appearance Models, Third Edition. Mark D. Fairchild. © 2013 John Wiley & Sons, Ltd. Published 2013 by John Wiley & Sons, Ltd
    
    % Normal viewing parameters used (may differ depending on stimulus)
    % -----
    % Main Parameters
    % xyzw = [95.04, 100, 108.89]; % reference white point - here D65
    % la = 20; %lumiance of adapting field. The adapting field is the total environment of the colour element considered, including the proximal field, the background and the surround, and extending to the limit of vision in all directions. ?CIECAM02 and recent developments? MR Luo and C. Li, in 'Advanced Color Image Processing and Analysis', 2002, ed. Christine Fernandez-Maloigne

    % yb = 17.16; %Y of background
    
    % Parameter decision table
    % para.c1 = 0.69;
    % para.Nc = 0.95;
    % para.F = 0.9;
        
    
    f = para(1); c = para(2); nc = para(3);

    % eq. 1 (from  Moroney, Fairchild, Hunt, Li, Luo and Todd Newman (2002)
    % 'The CIECAM02 Color Appearance Model')
    
    k = 1/(5*la+1);
    
    % eq. 2
        
    fl = 0.2*(k^4)*la + 0.1*((1-k^4)^2)*((5*la)^(1/3));
 
    % eq. 3
    
    n = yb/xyzw(2);
    
    % eq. 4
    
    ncb = 0.725*(1/n)^0.2;
    nbb = ncb;
    
    % eq. 5
    
    z = 1.48+sqrt(n);

    % Section: 'Chromatic Adaptation'
    
    % eq. 7

    M02 = [0.7328 0.4296 -0.1624; -0.7036 1.6975 0.0061; 0.0030 0.0136 0.9834];

    % eq. 6
    
    rgb = M02*xyz';
    rgbw = M02*xyzw';

    % Degree of adaptation

    % eq. 8
    
    D = f*(1-(1/3.6)*exp((-la-42)/92));

    % Start 'full' chromatic adaptation

    % eq. 9
    
    rgbc(1,1) = (xyzw(2)*(D/rgbw(1)) + (1 - D))*rgb(1);
    rgbc(2,1) = (xyzw(2)*(D/rgbw(2)) + (1 - D))*rgb(2);
    rgbc(3,1) = (xyzw(2)*(D/rgbw(3)) + (1 - D))*rgb(3);

    rgbwc(1,1) = (xyzw(2)*(D/rgbw(1)) + (1 - D))*rgbw(1);
    rgbwc(2,1) = (xyzw(2)*(D/rgbw(2)) + (1 - D))*rgbw(2);
    rgbwc(3,1) = (xyzw(2)*(D/rgbw(3)) + (1 - D))*rgbw(3);

    % eq. 12
    
    MH = [0.38971 0.68898 -0.07868; -0.22981 1.18340 0.04641; 0.0 0.0 1.0];

    % eq. 11
    
    Minv = [1.096124 -0.278869 0.182745; 0.454369 0.473533 0.072098; -0.009628 -0.005698 1.015326];

    % eq. 10  

    rgbp = MH*Minv*rgbc;
    rgbpw = MH*Minv*rgbwc;
    
    % Section: 'Non-linear response compression' 

    % eq. 13
    
    rgbpa(1,1) = (400*(fl*rgbp(1)/100).^0.42)./(27.13+(fl*rgbp(1)/100).^0.42)+0.1;
    rgbpa(2,1) = (400*(fl*rgbp(2)/100).^0.42)./(27.13+(fl*rgbp(2)/100).^0.42)+0.1;
    rgbpa(3,1) = (400*(fl*rgbp(3)/100).^0.42)./(27.13+(fl*rgbp(3)/100).^0.42)+0.1;

    rgbpwa(1,1) = (400*(fl*rgbpw(1)/100)^0.42)/(27.13+(fl*rgbpw(1)/100)^0.42)+0.1;
    rgbpwa(2,1) = (400*(fl*rgbpw(2)/100)^0.42)/(27.13+(fl*rgbpw(2)/100)^0.42)+0.1;
    rgbpwa(3,1) = (400*(fl*rgbpw(3)/100)^0.42)/(27.13+(fl*rgbpw(3)/100)^0.42)+0.1;

    % Section: 'Perceptual Attribute Correlates'
    
    % Opponent Color Dimensions

    % eq. 14
    
    a = rgbpa(1) - 12*rgbpa(2)/11 + rgbpa(3)/11;
    
    % eq. 15
    
    b = (1/9)-(rgbpa(1) + rgbpa(2) - 2*rgbpa(3));

    % Hue angle
    
    % eq. 17

    h = atand(b/a);

    % eccentricity factor
    
    % eq. 18
    
    %e = ...

    % Hue composition
    
    % eq. 19
    
    %H = ...
    
    % Preliminary magnitude
    
    % eq. 16
    
    %t = (e*(a^2+b^2)^0.5) / (rgbpa(1)+rgbpa(2)+(21/20)*rgbpa(3));
    
    % Achromatic response
    
    % eq. 20

    A = (2*rgbpa(1) + rgbpa(2) + rgbpa(3)/20 - 0.305)*nbb;
    Aw = (2*rgbpwa(1) + rgbpwa(2) + rgbpwa(3)/20 - 0.305)*nbb;

    % Lightness
    
    % eq. 21
    
    J = 100*(A/Aw)^(c*z);

    % Brightness
    
    % eq. 22

    Q = (4/c)*(sqrt(J/100))*((Aw + 4)*fl^0.25);

    % Chroma
    
    % eq. 23
    
    %C = ...

    % Colorfulness

    % eq. 24
    
    M = c*fl^0.25;

    % Saturation

    % eq. 25
    
    s = 100*sqrt(M/q);
    
    % Cartesian Representations

    % eq. 26
    
    % ...
    
    % eq. 27

    % ...
    
end



