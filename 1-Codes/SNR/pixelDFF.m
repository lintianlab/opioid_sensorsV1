function [dff] = pixelDFF(before, after, ind_nonzerofirst)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[m n]= size(before);
dff=zeros(m, n);
for i=1:length(ind_nonzerofirst)
    ind=ind_nonzerofirst(i);
    curval=(after(ind)-before(ind))./before(ind);
    dff(ind)=curval;
end
end

