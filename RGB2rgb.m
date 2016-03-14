function normalized_rgb = RGB2rgb(image)

image = im2double(image);

R = image(:,:,1);
G = image(:,:,2);
B = image(:,:,3);

sum = R+G+B;

r = R./sum;
g = G./sum;
b = B./sum;

normalized_rgb = cat(3,r,g,b);

end