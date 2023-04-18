function [BWfinal] = edgelogSegv2(im, ff, dil)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[~,threshold] = edge(im,'log');
fudgeFactor = ff; %change from 0.5 change from 1.2
BWs = edge(im,'log',threshold * fudgeFactor);
se90 = strel('line', dil, 90); %%%change from 4
se0 = strel('line', dil, 0); 
BWsdil2 = imdilate(BWs, [se90 se0]);
seD = strel('diamond',1);
BWfinal = imerode(BWsdil2,seD);
BWfinal = imerode(BWfinal,seD);
%imshow(BWfinal)
%BWC= bwareaopen(BWsdil, 300);
%figure, imshow(BWfinal)
end



