function out = interpolation2(xx, yy, x, method)

if nargin < 4
    method = 'pchip';
end

if (length(xx) > 1)
    if ( method == 'linear' )
        xx = [-10^9 xx 10^9];
        yy = [yy(1) yy yy(end)];
    end
    out = interp1(xx, yy, x, method);
else
    out = yy * ones(size(x));
end