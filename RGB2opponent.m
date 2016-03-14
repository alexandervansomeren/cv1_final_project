function opponent = RGB2opponent(image)

image = im2double(image);

R  = image(:,:,1);
G  = image(:,:,2);
B  = image(:,:,3);

O1 = (R-G)./sqrt(2);
O2 = (R+G-2*B)./sqrt(6);
O3 = (R+G+B)./sqrt(3);

opponent = cat(3,O1,O2,O3);

end