function out = OutputPowerAtERBNumbers025(InputLevels, f, in)

ERBc = 1.75:0.25:39;
fc = ERBnumber2frequency(ERBc);
Excitation = zeros( length(in), length(ERBc) ); 

for i=1:length(ERBc)

    pu = get_p(fc(i)) .* ones(1,length(in));
    pl = get_pl(fc(i),InputLevels);
    p  = pu .* ( f >= fc(i) ) + pl .* ( f < fc(i) ); 
    g = ( f - fc(i) ) ./ fc(i); 

    ExcitationOfThisERBc = get_W(p,g) .* (g <= 4 ) .* 10.^(in./10);
    Excitation((1:length(in)),i) = ExcitationOfThisERBc(1:length(in));
end

if ( size(Excitation,1) > 1 ) 
    Excitation = sum(Excitation);
end

Excitation = 10 .* log(Excitation) ./ log(10);

out = Excitation;