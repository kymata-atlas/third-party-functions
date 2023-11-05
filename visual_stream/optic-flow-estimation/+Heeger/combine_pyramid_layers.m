% combine

final_contour = [];

G4 = contour_G4*20;
G3 = contour_G3*10;
G2 = contour_G2*6.8;
G1 = contour_G1*5;
    

for i = 1:length(G1)
    

    if ( abs(G4(i)) >= 4.8 )
        thisguess = G4(i);
    elseif ( abs(G3(i)) >= 3)
        thisguess = G3(i)  ;  
    elseif ( abs(G2(i)) >= 2)
        thisguess = G2(i)  ;  
    else
        thisguess = G1(i);
    end
    
    final_contour = [final_contour thisguess];
end