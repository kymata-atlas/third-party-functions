function out = frequency2ERBnumber ( in )

% convert Hz to Cam

out = 21.366 * log(0.004368 * in + 1) / log(10);