function out = ERBnumber2frequency( in )

% convert Cam to Hz

out = ( 10 .^ ( in / 21.366 ) - 1 ) / 0.004368;