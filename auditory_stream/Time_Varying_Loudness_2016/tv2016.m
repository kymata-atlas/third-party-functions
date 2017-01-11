function [Loudness, ShortTermLoudness, LongTermLoudness] = tv2016( filenameSound, dBMax, filenameFilter, s, Fs )

% calculate loudness according to Moore, Glasberg & Schlittenlacher (2016)
% developed from:
% Glasberg & Moore (2002): time-varying part
% Moore & Glasberg (2007): binaural stationary loudness of each segment
% ANSI S3.4-2007: basis for Moore & Glasberg (2007)
%
% pass any dummy filename if you want to analyse signal s with two column
% vectors and sample rate Fs

addpath(genpath('functions'))
addpath(genpath('specific loudness'))

if nargin < 5
    [s, Fs] = ReadAndResample( filenameSound );
end

if ( Fs ~= 32000 )
    s = resample( s, 32000, Fs );
    Fs  = 32000;
end

% tic

s = SoundField2Cochlea(s, filenameFilter);

[InstantaneousSpecificLoudnessLeft, InstantaneousSpecificLoudnessRight] = FilteredSignal2MonauralInstantaneousSpecificLoudness( s, Fs, dBMax );

ShortTermSpecificLoudnessLeft  = InstantaneousSpecificLoudness2ShortTermSpecificLoudness( InstantaneousSpecificLoudnessLeft );
ShortTermSpecificLoudnessRight = InstantaneousSpecificLoudness2ShortTermSpecificLoudness( InstantaneousSpecificLoudnessRight );

ShortTermLoudnessLeft  = zeros( size( ShortTermSpecificLoudnessLeft, 1 ), 1 );
ShortTermLoudnessRight = zeros( size( ShortTermSpecificLoudnessRight, 1 ), 1 );

for i = 1:size( ShortTermSpecificLoudnessLeft, 1 )
    [~, ShortTermLoudnessLeft(i), ShortTermLoudnessRight(i)] = MonauralSpecificLoudness2BinauralLoudness025( ShortTermSpecificLoudnessLeft(i,:), ShortTermSpecificLoudnessRight(i,:) );
end

LongTermLoudnessLeft   = ShortTermLoudness2LongTermLoudness( ShortTermLoudnessLeft );
LongTermLoudnessRight  = ShortTermLoudness2LongTermLoudness( ShortTermLoudnessRight );

LongTermLoudness = LongTermLoudnessLeft + LongTermLoudnessRight;

Loudness = max(LongTermLoudness);

ShortTermLoudness = ShortTermLoudnessLeft + ShortTermLoudnessRight;


% toc

figure;
plot( 0:(length(ShortTermLoudness)-1), ShortTermLoudness, 'b-' );
hold on;
plot( 0:(length(LongTermLoudness)-1), LongTermLoudness, 'r-' );
xlabel( 'time [ms]');
ylabel( 'Loudness [sone]');
legend( 'Short-term loudness','Long-term loudness' );

strOutputFilename = [filenameSound ' ' num2str( dBMax ) ' dB calibration level TVL 2016.txt'];
fid = fopen([strOutputFilename],'wt+');
fprintf(fid,'1st column: time in milliseconds\n');
fprintf(fid,'2nd column: short-term loudness in sone\n');
fprintf(fid,'3rd column: short-term loudness level in phon\n');
fprintf(fid,'4th column: long-term loudness in sone\n');
fprintf(fid,'5th column: long-term loudness level in phon\n\n');
fprintf(fid,'   time   short-t. loudness    long-t. loudness\n');
fprintf(fid,'     ms      sone      phon      sone      phon\n');
for i=1:length( LongTermLoudness )
    fprintf(fid,'%7.0f %9.2f %9.1f %9.2f %9.1f\n',i-1, ShortTermLoudness(i), Sone2PhonTV2015(ShortTermLoudness(i)), LongTermLoudness(i), Sone2PhonTV2015(LongTermLoudness(i)) );
end
fclose(fid);
