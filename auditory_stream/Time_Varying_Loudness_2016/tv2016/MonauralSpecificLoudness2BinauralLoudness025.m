function [Loudness, LoudnessLeft, LoudnessRight] = MonauralSpecificLoudness2BinauralLoudness025( SpecificLoudnessLeftMon, SpecificLoudnessRightMon )

g = -18:0.25:18; 
W = exp( -(0.08 .* g) .^2 );

SpecificLoudnessLeftSmoothed  = conv(W,SpecificLoudnessLeftMon)/sum(W);
SpecificLoudnessLeftSmoothed  = SpecificLoudnessLeftSmoothed(73:222);   
SpecificLoudnessRightSmoothed = conv(W,SpecificLoudnessRightMon)/sum(W); 
SpecificLoudnessRightSmoothed = SpecificLoudnessRightSmoothed(73:222);

SpecificLoudnessLeftSmoothed  = SpecificLoudnessLeftSmoothed + 10^-13;
SpecificLoudnessRightSmoothed = SpecificLoudnessRightSmoothed + 10^-13;

p = 1.5978;

InhibLeft = 2 ./ ( 1 + ( sech( SpecificLoudnessRightSmoothed ./ SpecificLoudnessLeftSmoothed ) ) .^ p ); 
InhibRight = 2 ./ ( 1 + ( sech( SpecificLoudnessLeftSmoothed ./ SpecificLoudnessRightSmoothed ) ) .^ p );

SpecificLoudnessLeft = SpecificLoudnessLeftMon ./ InhibLeft; 
SpecificLoudnessRight = SpecificLoudnessRightMon ./ InhibRight;            

LoudnessLeft = sum(SpecificLoudnessLeft) / 4;   
LoudnessRight = sum(SpecificLoudnessRight) / 4;

Loudness = LoudnessLeft + LoudnessRight;  