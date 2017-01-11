function [InstantaneousSpecificLoudnessLeft, InstantaneousSpecificLoudnessRight] = FilteredSignal2MonauralInstantaneousSpecificLoudness(signal, Fs, dBMax)

addpath(genpath('functions'))

% calculates instantaneous specific loudness over time (left, right)
% signal and sampling rate Fs must be passed. Fs must be 32 kHz, reading of the wav file
% and resampling must be done before calling Signal2Loudness. signal must
% contain two columns.
% dBMax is the RMS SPL of a sinusoid with a peak amplitude of 1.

npts               = Fs / 1000 * 64;                     % points for FFT, 2048
nSegmentDuration   = 1;                                  % duration of segment in ms
nSamplesPerSegment = Fs / 1000 * nSegmentDuration;       % 32
nSegmentsInSignal  = floor( (length(signal) - npts) / Fs * 1000 / nSegmentDuration );

% Hann windows for 6 FFTs; 1st column 64 ms, 6th column 2 ms
wHann = zeros(npts,6);
for i = 1:6
    wHann(:,i) = [ zeros( (1 - 1/2^(i-1) ) / 2 * npts, 1 ); hann( npts / 2^(i-1) ); zeros( (1 - 1/2^(i-1) ) / 2 * npts, 1 ) ];
end

% indices which shall be used from the ffts, 20-80 Hz from the first...
vLimitingF = [20 80 500 1250 2540 4050 15000]; 
vLimitingIndices = zeros(1,7);
for i=1:7
    vLimitingIndices(i) = floor( vLimitingF(i) / (Fs/npts) ) + 1;
end

InstantaneousSpecificLoudnessLeft  = zeros(nSegmentsInSignal + 1, length(1.75:0.25:39) );
InstantaneousSpecificLoudnessRight = zeros(nSegmentsInSignal + 1, length(1.75:0.25:39) );

%% loop: window segment, apply stationary loudness model, move window
%% by 1 ms

% tic
for iSegment = 0:nSegmentsInSignal
%     
%     if ( mod(iSegment,50) == 0 )
%         disp([num2str(iSegment) ' ms of ' num2str(nSegmentsInSignal) ' ms analyzed'])
%     end
    
    [fLeftRelevant, LLeftRelevant, fRightRelevant, LRightRelevant] = SignalSegment2Spectrum(signal( (iSegment*nSamplesPerSegment+1) : (iSegment*nSamplesPerSegment+npts), : ), Fs, dBMax, wHann, vLimitingIndices);
    
    if ( isempty( LLeftRelevant ) )
        SpecificLoudnessLeft = zeros(1,150);
    else
        ExcitationLevelsLeft = Spectrum2ExcitationPattern025(fLeftRelevant', LLeftRelevant');
        SpecificLoudnessLeft = Excitation2SpecificLoudnessBinaural025( ExcitationLevelsLeft );
    end
    if ( isempty( LRightRelevant ) )
        SpecificLoudnessRight = zeros(1,150);
    else
        ExcitationLevelsRight = Spectrum2ExcitationPattern025(fRightRelevant', LRightRelevant');
        SpecificLoudnessRight = Excitation2SpecificLoudnessBinaural025( ExcitationLevelsRight );
    end
%     [ ThisSpecificLoudness, ThisLoudness ] = MonauralSpecificLoudness2BinauralSpecificLoudness025( SpecificLoudnessLeft, SpecificLoudnessRight );
    InstantaneousSpecificLoudnessLeft(iSegment+1,:) = SpecificLoudnessLeft;
    InstantaneousSpecificLoudnessRight(iSegment+1,:) = SpecificLoudnessRight;
%     InstantaneousLoudness(iSegment+1)         = ThisLoudness;
end
% toc

