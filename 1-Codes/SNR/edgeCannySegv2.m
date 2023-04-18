function [BWfinal] = edgeCannySegv2(dat, ff, dil)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
[~,threshold] = edge(dat,'Canny');
fudgeFactor = ff; %2
BWs = edge(dat,'Canny',threshold * fudgeFactor);
se90 = strel('line',dil, 90); %8 default
se0 = strel('line', dil, 0);  %8 default
BWsdil2 = imdilate(BWs, [se90 se0]);
seD = strel('diamond',1);
BWfinal = imerode(BWsdil2,seD);
BWfinal = imerode(BWfinal,seD);
%figure, imshow(BWfinal)
%BWC= bwareaopen(BWsdil, 300);

%figure, imshow(BWsdil2)
end

