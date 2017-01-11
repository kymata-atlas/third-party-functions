function out = get_pl(f,X)

out = get_p(f) - 0.35 .* ( get_p(f) ./ get_p(1000) ) .* ( X - 51 );