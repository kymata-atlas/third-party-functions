function out = Excitation2SpecificLoudnessBinaural025( ExcitationLevels )

% version for TVL 2016 based on ANSI S3.4-2007 and Moore & Glasberg (2007)
% Calculate the specific loudness out of excitation patterns at 0.25-ERB
% steps, using the constant C of the model that accounts for binaural
% loudness

% ERB-scale
ERBc = 1.75:0.25:39;
fc = ERBnumber2frequency(ERBc);

% Parameters
C = 0.06267;
C = 0.063026;
C = 0.0631;
G = get_G_TVL(fc);
Alpha = get_Alpha(fc);
A = get_A(fc);

% Excitation and threshold in linear power units
Excitation = 10 .^ ( ExcitationLevels ./ 10 );
Threshold = ExcitationThreshold_TVL(fc);
Threshold = 10 .^ ( Threshold ./ 10 );

% 3 ranges - calculate all for each case (computationally cheap)
N1 = C .* ( ( G .* Excitation + A ) .^ Alpha - A .^Alpha );
N2 = C .* ( 2 .* Excitation ./ ( Excitation + Threshold ) ) .^ 1.5 .* ( ( G .* Excitation + A ) .^ Alpha - A .^Alpha );
N3 = C .* ( Excitation ./ 1.0707 ) .^ 0.2;

out = ( Excitation > Threshold & Excitation < 10^10 ) .* N1 + ( Excitation <= Threshold ) .* N2 + (Excitation >= 10^10 ) .* N3;

% figure;
% hold on;
% semilogy(ExcitationLevels, out);
% xlim([0 110]);
% ylim([0.005 10]);