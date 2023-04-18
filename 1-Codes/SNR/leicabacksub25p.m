function [im_back] = leicabacksub25p(image)
imaged=double(image);
[rows columns]=size(imaged);
matofmeans= zeros(rows, columns);
indices=[-2 -1 0 1 2];
neighbors=zeros(5, 5); %5 by 5 matrix to keep values around centerpixel
    for row=3:rows-2
        for col=3:columns-2
            for i=1:5
                first= indices(i);
                for j=1:5
                    second=indices(j);
                    if first==0 & second==0
                    else
                      val=imaged(row+first, col+second);
                      neighbors(3+first, 3+second)= val;
                    end
                end
            end
            neighbors(3,3)= imaged(row,col);
            meanpix= mean(neighbors(:));
            matofmeans(row, col)=meanpix;
        end
    end
    matofmeansr=matofmeans(3:rows-2, 3:columns-2);
    minmeanpix=min(matofmeansr(:));
    im_back=imaged-minmeanpix; %note that there is clipping of negative values to 0
end

