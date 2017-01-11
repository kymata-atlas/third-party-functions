function [out Fs] = ReadAndResample( filename )

[s, Fs] = wavread( filename );

if ( Fs == 32000 )
    out = s;
else
    out = resample( s, 32000, Fs );
    Fs  = 32000;
end