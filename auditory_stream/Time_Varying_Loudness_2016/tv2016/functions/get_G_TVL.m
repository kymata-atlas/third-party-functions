function out = get_G_TVL(f)

% low level gain of the cochlear amplifier relative to the gain at 500 Hz

LinearThreshold = 10.^( ExcitationThreshold_TVL(f) ./ 10 );
out = 10^(3.63/10) ./ LinearThreshold;