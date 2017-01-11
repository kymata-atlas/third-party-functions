function out = get_p(f)

% parameter for slope of auditory filter

out = 4 .* f ./ get_ERB(f);