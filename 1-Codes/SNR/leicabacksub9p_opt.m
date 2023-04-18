function [im_back] = leicabacksub9p_opt(image)
imaged=double(image);
[rows columns]=size(imaged);
kernel=ones(3,3)/9;
C=conv2(imaged, kernel, 'same');
Cfinal=C(2:rows-1, 2:columns-1);
minmeanpix=min(Cfinal(:));
im_back=imaged-minmeanpix;

end

