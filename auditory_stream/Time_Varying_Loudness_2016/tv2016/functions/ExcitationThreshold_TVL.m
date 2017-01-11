function out = ExcitationThreshold( f )
         
Above500 = 3.63;
Threshold = [ 50    63    80    100   125   160   200  250  315  400  500  630  750  800  1000;
              28.18 23.90 19.20 15.68 12.67 10.09 8.08 6.30 5.30 4.50 3.63 3.63 3.63 3.63 3.63 ];

Below500 = interpolation2(Threshold(1,:), Threshold(2,:), f, 'linear');
          
out = ( f < 500 ) .* Below500 + ( f >= 500 ) .* Above500;