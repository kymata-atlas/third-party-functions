clc
clear all
close all

% *************************************************************************
% Estimation of optic flow for the image sequence "Translation Sphere".
% This sequence contains only one layer of motion and, thus, I only use
% methods that support the estimation of one layer motion. I do not
% compute estimation errors. For a comprehensive evaluation of optic flow
% methods see http://vision.middlebury.edu/flow/eval/.
% For the evaluation of these algorithms I created another image sequence
% "TranslationSphere" because known benchmark sequences typically do not 
% provide enough frames for all methods.
%
% Attention: This script may take tens of seconds to finish!
% 
%   Copyright (C) 2013  Florian Raudies, 05/16/2013, Boston University.
%   License, GNU GPL, free software, without any warranty.
% *************************************************************************

% Some algorithms would require more mechanisms to detect multiple speeds,
% more directions, etc. To avoid long runtimes and for educational reasons 
% I kept the scripts rather simple. Thus, results give only a guideline as
% to how good these methods are.
% algoIndex = 1; % Select the algorithm to run with this index.
% %                   Authors         frames        no
% AlgoNameFrames = {{'AdelsonBergen', 'all'}, ... % 1
%                   {'Farnebaeck',    'all'}, ... % 2
%                   {'FleetJepson',   'all'}, ... % 3
%                   {'Heeger',        'all'}, ... % 4
%                   {'HornSchunck',   'two'}, ... % 5
%                   {'LucasKanade',   'two'}, ... % 6
%                   {'MotaEtAl',      'all'}, ... % 7
%                   {'Nagel',         'all'}, ... % 8
%                   {'OtteNagel',     'all'}, ... % 9
%                   {'UrasEtAl',      'all'}};    % 10
              
gaussLayer = 4;
contour = [NaN NaN NaN NaN NaN NaN NaN NaN];
for frame = 1:350%120*400  % 1:55

% Load the image sequence.
ImgSeq        = stream_our_stim(frame, 15, 120, 'greyscale', gaussLayer);

% if strcmp(AlgoNameFrames{algoIndex}{2},'two'),
%     ImgSeq = ImgSeq(:,:,11:12);
% end

warning_state = warning;
warning('off', 'all');


% Heeger

%[Dx, Dy, L] = Heeger.estimateOpticFlow2D(double(ImgSeq));
%warning(warning_state);
% contour = [contour, nanmean(Dx(:))];

% AdelsonBergen_ME

[Dx, Dy, or_me_plus, or_me_minus, op_me] = AdelsonBergen.estimateOpticFlow2D_energy(double(ImgSeq));
contour = [contour, nanmean(op_me(:))];

% Heeger_ME

%if (mod(frame,2))
%    [motion_energy] = Heeger.estimateOpticFlow2D_energy(double(ImgSeq));
%    x = motion_energy(:,:,3,1);
%    avg_motion_energy = mean(x(:));
%    contour = [contour, avg_motion_energy];
%else
%    contour(end+1) = contour(end);
%end

% PRINT

% imagesc(Dx);
% title(sprintf(['Adelson-Bergen, default settings, frame ', num2str(first_frame)]))
% map = [ 1 0 0
%        0.9 0 0
%        0.8 0 0
%        0.7 0 0
%        0.6 0 0
%        0.5 0 0
%        0.4 0 0
%        0.3 0 0
%        0.2 0 0
%        0.1 0 0
%        0 0 0
%        0 0.1 0
%        0 0.2 0
%        0 0.3 0
%        0 0.4 0
%        0 0.5 0
%        0 0.6 0
%        0 0.7 0
%        0 0.8 0
%        0 0.9 0
%        0 1 0 ];
% colormap(map);
% caxis([-1 1]);
% saveas(gcf,['output/Dx_Adelson-Bergen_frame_', num2str(first_frame), '.jpg'])

end
