function Nlt = ShortTermLoudness2LongTermLoudness( Nst )

aA = 0.01;
aR = 0.0005;

Nlt = zeros( size(Nst) );

Nlt(1) = AGCNextFrame( 0, Nst(1), aA, aR );
for i=2:length(Nst)
    Nlt(i) = AGCNextFrame( Nlt(i-1), Nst(i), aA, aR );
end