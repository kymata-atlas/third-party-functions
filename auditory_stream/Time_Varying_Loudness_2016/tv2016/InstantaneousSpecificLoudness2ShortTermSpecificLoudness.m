function [ShortTermSpecificLoudness, ShortTermLoudness] = InstantaneousSpecificLoudness2ShortTermSpecificLoudness( InstantaneousSpecificLoudness )

aA = 0.045;
aR = 0.02;

ShortTermSpecificLoudness = zeros( size(InstantaneousSpecificLoudness) );

ShortTermSpecificLoudness(1,:) = AGCNextFrameOfVector( zeros(1,150) , InstantaneousSpecificLoudness(1,:), aA, aR );
for i=2:size( InstantaneousSpecificLoudness, 1 )
    ShortTermSpecificLoudness(i,:) = AGCNextFrameOfVector( ShortTermSpecificLoudness(i-1,:), InstantaneousSpecificLoudness(i,:), aA, aR );
end

ShortTermLoudness = sum( ShortTermSpecificLoudness, 2 ) / 4;