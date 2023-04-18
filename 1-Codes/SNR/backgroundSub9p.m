function [stack_b] = backgroundSub9p(stack)
%Adapted from ImageAnalyst's localBinaryPattern
%%%this subtracts background from everything
[m n]=size(stack);
stack_b=cell(m,2); %populate new cell array to hold background subtracted images
stack_b(:,2)=stack(:,2); %populate new array with names
for i=1:m
    curstack=double(stack{i, 1});
    [rows columns]=size(curstack);
    %if median(curstack(:))==0
    %    stack_b{i,1}=curstack;
    %else
    matofmeans= zeros(rows, columns);
    for row=2:rows-1
        for col=2:columns-1
            centerPixel= curstack(row,col);
            pixel7=curstack(row-1, col-1);
            pixel6=curstack(row-1, col);
            pixel5=curstack(row-1, col+1);
            pixel4=curstack(row, col+1);   
            pixel3=curstack(row+1, col+1) ;   
            pixel2=curstack(row+1, col) ;     
            pixel1=curstack(row+1, col-1) ;    
            pixel0=curstack(row, col-1) ;
            squarepix= [centerPixel, pixel7, pixel6, pixel5, ...
                          pixel4, pixel3, pixel2, pixel1, pixel0];
            meanpix= mean(squarepix);
            matofmeans(row, col)=meanpix;
            
        end
    end
    matofmeansr=matofmeans(2:rows-1, 2:columns-1);
    minmeanpix=min(matofmeansr(:));
    cur.back=curstack-minmeanpix; %note that there is clipping of negative values to 0
    stack_b{i,1}=cur.back;
end
%%%%%%%%%%%%%%%%%%
end