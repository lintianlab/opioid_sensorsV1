function [T, dffs, nonzeromask, snrs] = timeconf_gui(stack, framebefore1, framebefore2, backgroundsub, ...
                                            register, edgemethod, ff, dil)
%%% Time course for app
%%% Nikki Tjahjono
%%% 12/31/20
%edgemethod either "Canny", "Log", or "Sobel" if nothing specified, does
%Canny!


beforeframes=framebefore1:framebefore2; %Which frames are before ligand addition?

%[files, path] = uigetfile('.lsm', 'Select All Image Files', 'MultiSelect','on');

%data= bfopen(fullfile(path,files));
%stack=data{1,1}; %cell with all the planes
[j k]= size(stack);
    mediandff=zeros(j, 1);
    if register==0
        registered= zeros(j, 1);
    else
        registered=ones(j, 1);
    end
    theta= zeros(j, 1);
    allstacks=1:j;
    maskdff= zeros(j, 1);
    combmaskdff=zeros(j,1);
    indivmaskdff=zeros(j,1);
    mediansnr=zeros(j,1);
    T=table(allstacks', mediandff, registered, maskdff, combmaskdff, indivmaskdff, mediansnr);
  
    
%stack{i, 1} %i selects for individual stack

if backgroundsub==1
    stack_b=medbackgroundSub(stack);
else
    stack_b=stack;
end

[m n]=size(stack_b{1,1});
f0=zeros(m, n, length(beforeframes));
for a=1:length(beforeframes)
%%%%take average of before glut frames
    currframe=beforeframes(a);
    f0(:,:,a)=stack_b{currframe,1};
end
f0mean=mean(f0,3);
sqrtf0mean= sqrt(f0mean); %use for SNR calculation


%%%calculate all frames

dffs=zeros(m, n, j);
nonzeromask=cell(j,1);
nonzeromask_curr=cell(j,1);
for b=1:j
if register==1 %if you want to register
    try
        tryreg=featureRegistration(f0mean,stack_b{b, 1});
        if abs(tryreg.theta)>2 %if theta is extreme, void the registration
            currreg=stack_b{b, 1};
            T{b,3}=0;
        else %if theta isn't extreme, just use it and report value
           T{b,4}=tryreg.theta;
           currreg=tryreg.im;
        end
    catch %if registration not successful (throw error), do not register
        currreg=stack_b{b, 1};
        T{b, 3}=0;
    end
else
    currreg=stack_b{b, 1};
end
    if edgemethod== "Log"
        curr_mask=edgelogSegv2(currreg, ff, dil);
    elseif edgemethod== "Sobel"
        curr_mask=edgeSobelSegv2(currreg, ff, dil);
    else
        curr_mask=edgeCannySegv2(currreg, ff, dil);
    end
    
    
    before=f0mean.*curr_mask;
    curr_seg=double(currreg).*curr_mask;
    ind_nonzero_curr=find(curr_seg>0);
    ind_nonzero=find(before>0);
    currdffmat=pixelDFF(before, curr_seg, ind_nonzero);
    dffs(:,:,b)=currdffmat;
    snrframe=currdffmat.*sqrtf0mean;
    snrs(:,:,b)=currdffmat.*sqrtf0mean;
    nonzeromask{b}=ind_nonzero;
    nonzeromask_curr{b}=ind_nonzero_curr;
    currdff=median(currdffmat(ind_nonzero)); %changed to median
    T{b, 2}=currdff;
    T{b, 7}=median(snrframe(ind_nonzero));
    wholemask=(mean(curr_seg(ind_nonzero))/mean(before(ind_nonzero)))-1;
    T{b, 4}=wholemask;
    
    %%%add this for indivmaskdff
    if edgemethod== "Log"
        before_mask=edgelogSegv2(f0mean, ff, dil);
    elseif edgemethod== "Sobel"
        before_mask=edgeSobelSegv2(f0mean, ff, dil);
    else
        before_mask=edgeCannySegv2(f0mean, ff, dil);
    end
    before_beforeseg=f0mean.*before_mask;
    ind_nonzero_beforeseg=find(before_beforeseg>0 & before_beforeseg<255);
    indivmaskdffmedian=(median(curr_seg(ind_nonzero))/median(before_beforeseg(ind_nonzero_beforeseg)))-1;
    T{b, 6}=indivmaskdffmedian;
    %%%
    
end
    %combine mask and then apply to each
    maskcomb=union(nonzeromask_curr{1}, nonzeromask_curr{2});
    for i=3:j
        maskcomb=union(maskcomb, nonzeromask_curr{i});
    end
    total=m*n;
    list=1:total;
    backgroundmask= setxor(list, maskcomb);
    
    meanscombmask=zeros(j,1);
    for b=1:j
        currstack=stack{b,1};
        %find background of current stack
        background=currstack(backgroundmask);
        mean_back=mean(double(background));
        sd_back=std(double(background));
        backgroundlevel=mean_back+sd_back;
        backgroundlevel=0;
        %
        abovebackind=find(currstack(maskcomb)>backgroundlevel);
        meanscombmask(b,1)=mean(currstack(abovebackind)); 
    end
    beforefscomb=zeros(length(beforeframes),1);
    for a=1:length(beforeframes)
        %%%%take average of before glut frames
        currframe=beforeframes(a);
        beforefscomb(a)=meanscombmask(currframe);
    end
    

    beforemeancomb=mean(beforefscomb);
    T{:, 5}=(meanscombmask./beforemeancomb)-1;
    
    
    
end

