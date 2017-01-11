function out = AGCNextFrame( dLastFrame, dThisInput, aA, aR )

if ( dThisInput > dLastFrame )                            % attack
    out = aA * dThisInput + ( 1 - aA ) * dLastFrame;
else                                                      %release
    out = aR * dThisInput + ( 1 - aR ) * dLastFrame;
end