function out = get_Alpha(f)

TableG2Alpha = [ -25.0   -20.0   -15.0   -10.0   -5.0    0.0;
                 0.26692 0.25016 0.23679 0.22228 0.21055 0.20000];

G = get_G_TVL(f);
G = 10 .* log(G) ./ log(10);

out = interpolation2(TableG2Alpha(1,:), TableG2Alpha(2,:), G,'linear');