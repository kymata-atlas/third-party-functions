function out = SoundField2Cochlea( s, filenameFilter )
    load(['transfer functions/' filenameFilter]);
    out(:,1) = conv( vecCoefficients, s(:,1) );
    if ( min( size(s) ) > 1 )
        out(:,2) = conv( vecCoefficients, s(:,2) );
    else
        out(:,2) = out(:,1);
    end
    out = out((1025):(end-1024),:);
%     out = [zeros(1024,2); s; zeros(1024,2)];  % without filter
end