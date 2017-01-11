function out = get_W(p,g)

g = abs(g);
out = ( 1 + p .* g ) .* exp( -1 .* p .* g );