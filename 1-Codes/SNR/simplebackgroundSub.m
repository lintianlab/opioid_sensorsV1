function [im_back] = simplebackgroundSub(image)
imaged=double(image);
val=median(imaged(:));
im_back=imaged-val; %note that there is clipping of negative values to 0
end

