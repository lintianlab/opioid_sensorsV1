%%% batch version of testimageanalysis_leica
% Nikki Tjahjono Updated 8/2/21
plothistogram=0 %set to 1 if you want to plot ALL dff histograms
dffimages=0 %set to 1 if you want to show ALL heatmaps
plotdfflim=2 %upper limit for dff heatmaps
savetable=1 %set to 1 if you want to save a table of your results. At
             %the end of the analysis, the program will prompt you to
             %select a folder to save "results.csv" to
backgroundsubmethod= "9pix" %NEW options are '25pix', '9pix', or 'simple' (fastest)
%%this block is for changing edge detection method; use timeconffull_exported to
%%determine best parameters
edgemethod="Canny" %options are "Canny", "Log", and "Sobel"
ff=1.5 %modify to change edge detection sensitivity
dil=4 %modify to change edge detection dilation factor
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
[filesbefore, pathbefore] = uigetfile('.TIF','Select All Image Files', 'MultiSelect','on');
[filesafter, pathafter] = uigetfile('.TIF','Select All Image Files', 'MultiSelect','on');
addpath('Dependencies_V2');

titles=filesbefore';
pixeldff=zeros(length(filesbefore), 1);
f0=zeros(length(filesbefore), 1);
maskdff=zeros(length(filesbefore), 1);
indivmaskdff=zeros(length(filesbefore), 1);
fpost=zeros(length(filesbefore), 1);
T=table(titles, f0, pixeldff, maskdff, indivmaskdff, fpost);


for f = 1:numel(filesbefore)
    after=imread(fullfile(pathafter, filesafter{f}));
    before=imread(fullfile(pathbefore, filesbefore{f}));
    if backgroundsubmethod=="simple"
        after_b=simplebackgroundSub(after);
        before_b=simplebackgroundSub(before);
    elseif backgroundsubmethod=="9pix"
        after_b=leicabacksub9p_opt(after);
       before_b=leicabacksub9p_opt(before);
    else
        after_b=leicabacksub25p_opt(after);
        before_b=leicabacksub25p_opt(before);
    end
    
    %after_b = imsharpen(after);
    %before_b = imsharpen(before);
    
           switch edgemethod
                case 'Canny' %where "Button1" is the name of the first radio button
                    after_mask=edgeCannySegv2(uint16(after_b), ff, dil);
                    before_mask=edgeCannySegv2(uint16(before_b),ff, dil);
                case 'Log'
                    after_mask=edgelogSegv2(uint16(after_b), ff, dil);
                    before_mask=edgelogSegv2(uint16(before_b),ff, dil);
                case 'Sobel'
                    after_mask=edgeSobelSegv2(uint16(after_b), ff, dil);
                    before_mask=edgeSobelSegv2(uint16(before_b),ff, dil);
                otherwise
            end


%%%

before_seg=double(before_b).*after_mask;
before_seg_2=double(before_b).*before_mask;
after_seg=double(after_b).*after_mask;


ind_nonzerobefore=find(before_seg_2>0);
ind_nonzeroafter=find(after_seg>0);
ind_nonzero10=find(before_seg>0 & after_seg>0);

dff=pixelDFF(before_seg, after_seg, ind_nonzero10);
afterglutdff=median(dff(ind_nonzero10)) ; %changed to median
T{f, 3}=afterglutdff;
T{f, 2}=mean(before_seg(ind_nonzero10));
T{f, 4}=(mean(after_seg(ind_nonzero10))/ ...
    mean(before_seg(ind_nonzero10)))-1;
T{f, 5}=(mean(after_seg(ind_nonzeroafter))/ ...
    mean(before_seg_2(ind_nonzerobefore))-1);
T{f, 6}=mean(after_seg(ind_nonzeroafter));
if dffimages==1
    figure
    colormap(jet(10));
    imagesc(dff)
    colorbar
    caxis([0 plotdfflim]) %change 2 to max DFF you want plotted for colorbar
    axis off;
end
if plothistogram==1
    figure
    hold on
    h1=histogram(dff(ind_nonzero10));
    h1.Normalization = 'probability';
    h1.BinWidth = 0.2;
    hold off 
end
fprintf('\n Image pair %d analysis done', f)
end
if savetable==1
    selpath=uigetdir("~/Documents/");
    writetable(T, strcat(selpath, "/", "results.csv"));
else
end
msgbox('Analysis complete')

