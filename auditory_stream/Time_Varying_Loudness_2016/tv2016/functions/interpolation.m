function out = interpolation(xx, yy, x)

out = interp1(xx, yy, x, 'cubic');