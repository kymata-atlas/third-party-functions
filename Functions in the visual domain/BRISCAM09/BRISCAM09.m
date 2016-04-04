function [a, b, an, bn, A, J ] = BRISCAM09(xyz, xyzw, la, yb, para)

    % This function is an implementation of the BRISCAM09 color appearance model
    % -------
    % Authors: Timo Kunkel and Erik Reinhard, A neurophysiology-inspired steady-state color appearance model, J. Opt. Soc. Am. A/Vol. 26, No. 4/April 2009
    % doi: 1084-7529/09/040776-7/

    % Input stream: The average X Y Z tristimulus values over the entire field of vision; the white reference X Y Z tristimulus values, and various 'viewing parameters' (per single frame window).
    
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

    % Section: 'Preliminary Computations'

    % eqs. 1-5
    
    k = 1/(5*la+1);
    fl = 0.2*(k^4)*la + 0.1*((1-k^4)^2)*((5*la)^(1/3));
    n = yb/xyzw(2);
    ncb = 0.725*(1/n)^0.2;
    Nbb = ncb;
    z = 1.48+sqrt(n);

    % eq 6

    D = f-((f/3.6)*exp((-la-42)/92));
    
    
    % Section: 'Chromatic Adaptation and Nonlinear Response Compression'

    MH = [0.38971 0.68898 -0.07868; -0.22981 1.18340 0.04641; 0.0 0.0 1.0];

    % eq. 7
    
    lms= MH*xyz';
    lmsw = MH*xyzw';

    % eq. 8a
    
    sigmaL = 27.13^(1/0.42)*(D*(lmsw(1)/100)+(1-D));
    
    % eq. 8b
        
    sigmaM = 27.13^(1/0.42)*(D*(lmsw(2)/100)+(1-D));
    
    % eq. 8c
    
    sigmaS = 27.13^(1/0.42)*(D*(lmsw(3)/100)+(1-D));
    
    % The nonlinear response compression 
    
    % eq. 9a-c
    
    lmsa(1,1) = (400*((fl*lms(1)/100)^0.42)/(((fl*lms(1)/100)^0.42)+(sigmaL^0.42)))+0.1;
    lmsa(2,1) = (400*((fl*lms(2)/100)^0.42)/(((fl*lms(2)/100)^0.42)+(sigmaM^0.42)))+0.1;
    lmsa(3,1) = (400*((fl*lms(3)/100)^0.42)/(((fl*lms(3)/100)^0.42)+(sigmaS^0.42)))+0.1;

    lmswa(1,1) = (400*((fl*lmsw(1)/100)^0.42)/(((fl*lmsw(1)/100)^0.42)+(sigmaL^0.42)))+0.1;
    lmswa(2,1) = (400*((fl*lmsw(2)/100)^0.42)/(((fl*lmsw(2)/100)^0.42)+(sigmaM^0.42)))+0.1;
    lmswa(3,1) = (400*((fl*lmsw(3)/100)^0.42)/(((fl*lmsw(3)/100)^0.42)+(sigmaS^0.42)))+0.1;             
            
    % Section: 'Computation of Appearance Correlates'
    
    % Achromaticity Response
    
    % eqs. 10 & 11
    
    A = Nbb*(4.19*lmsa(1)  + lmsa(2)  + 1.17*lmsa(3));
    AW = Nbb*(4.19*lmswa(1)  + lmswa(2)  + 1.17*lmswa(3));
    
    % Lightness

    % eq. 12

    J = 106.5*(A/AW)^(c*z); 
    
    % eq. 13

    % ...
    
    % eq. 14

    % ...
    
    % eq. 15
    
    % ...
    
    color_opponent_transform = [-15.4141 17.1339 -1.7198 ; -1.6010 -0.7467 2.3476];
    
    % eq. 16

    ab = color_opponent_transform * lmsa;

    a = ab(1);
    
    b = ab(2);
    
    % eq. 17
    
    h = atand(b/a);

    % eqs. 18-21
     
    rp = 0.6581*max([0, (cosd(9.1 - h))])^0.5390;
    gp = 0.9482*max([0, (cosd(167.0 - h))])^2.9435;
    
    yp = 0.9041*max([0, (cosd(90.9 - h))])^2.5251;
    bp = 0.7832*max([0, (cosd(268.4 - h))])^0.2886; 
    
    % Mapping into perceptual color opponent space
    
    % eqs. 22
    
    an = rp - gp;
    
    % eqs. 23
    
    bn = yp - bp;
    
    % eqs. 24
    
    ha = atand(bn/an);

end



