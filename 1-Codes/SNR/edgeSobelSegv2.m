function [BWfinal] = edgeSobelSegv2(dat, ff, dil)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[~,threshold] = edge(dat,'Sobel');
fudgeFactor = ff;
BWs = edge(dat,'Sobel',threshold * fudgeFactor);
se90 = strel('line',dil, 90); 
se0 = strel('line', dil, 0); 
BWsdil2 = imdilate(BWs, [se90 se0]);
seD = strel('diamond',1);
BWfinal = imerode(BWsdil2,seD);
BWfinal = imerode(BWfinal,seD);

%BWC= bwareaopen(BWsdil, 300);
%imshow(BWC)
%figure, imshow(BWfinal)
end

