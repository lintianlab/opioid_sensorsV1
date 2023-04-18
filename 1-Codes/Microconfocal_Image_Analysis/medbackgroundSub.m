function [stack_b] = medbackgroundSub(stack)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%%this subtracts background from everything
[m n]=size(stack);
stack_b=cell(m,2); %populate new cell array to hold background subtracted images
stack_b(:,2)=stack(:,2); %populate new array with names
for i=1:m
    curstack=double(stack{i, 1});
    %if median(curstack(:))==0
    %    stack_b{i,1}=curstack;
    %else
        val=median(curstack(:));
        cur.back=curstack-val; %note that there is clipping of negative values to 0
        stack_b{i,1}=cur.back;
    %end
end
%%%%%%%%%%%%%%%%%%
end

