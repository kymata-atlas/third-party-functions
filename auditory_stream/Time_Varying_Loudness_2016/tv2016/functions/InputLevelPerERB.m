function InputLevels = InputLevelPerERB(f,in)

ERBw = get_ERB(f);

InputLevels = zeros( length(in), length(in) );

for i = 1:length(in)
    p = get_p( f(i) );
    g = ( f - f(i) ) ./ f(i);
    InputLevelsOfThisTone = (g <= 4 ) .* get_W(p,g);
    InputLevelsOfThisTone = InputLevelsOfThisTone .* 10^(in(i)/10);
    InputLevels( i, (1:length(in)) ) = InputLevelsOfThisTone;
end

InputLevels = sum(InputLevels);

InputLevels = 10*log(InputLevels)./log(10);