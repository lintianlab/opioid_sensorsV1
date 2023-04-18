function [im_back] = leicabacksub25p_opt(image)

imaged=double(image);
[rows columns]=size(imaged);
kernel=ones(5,5)/25;
C=conv2(imaged, kernel, 'same');
Cfinal=C(3:rows-2, 3:columns-2);
minmeanpix=min(Cfinal(:));
im_back=imaged-minmeanpix;

end

