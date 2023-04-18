function [im_back] = leicabacksub9p(image)
imaged=double(image);
[rows columns]=size(imaged);
matofmeans= zeros(rows, columns);
    for row=2:rows-1
        for col=2:columns-1
            centerPixel= imaged(row,col);
            pixel7=imaged(row-1, col-1);
            pixel6=imaged(row-1, col);
            pixel5=imaged(row-1, col+1);
            pixel4=imaged(row, col+1);   
            pixel3=imaged(row+1, col+1) ;   
            pixel2=imaged(row+1, col) ;     
            pixel1=imaged(row+1, col-1) ;    
            pixel0=imaged(row, col-1) ;
            squarepix= [centerPixel, pixel7, pixel6, pixel5, ...
                          pixel4, pixel3, pixel2, pixel1, pixel0];
            meanpix= mean(squarepix);
            matofmeans(row, col)=meanpix;
            
        end
    end
    matofmeansr=matofmeans(2:rows-1, 2:columns-1);
    minmeanpix=min(matofmeansr(:));
    im_back=imaged-minmeanpix; %note that there is clipping of negative values to 0
end

