addpath metrics

H_over = '';
H_under = '';
H_result = '';

fileExt = '*.png';
Method_name = 'test';
save_dir ='';
file_name = fullfile(save_dir, strcat(Method_name, '.xlsx'));

fileso = dir(fullfile(H_over,fileExt)); 
filesu = dir(fullfile(H_under,fileExt)); 
filesr = dir(fullfile(H_result,fileExt)); 
len1 = size(fileso,1);

for i=1:len1;
    i
    fileNameo = strcat(H_over,fileso(i).name);
    imageo = imread(fileNameo);
    
    fileNameu = strcat(H_under,filesu(i).name);
    imageu = imread(fileNameu);
    
    fileNamer = strcat(H_result,filesr(i).name);
    imager = imread(fileNamer);

    Psnr(i) = metricsPsnr(imageo, imageu, imager);
    SF(i) = metricsSpatial_frequency(imageo, imageu, imager);
    CS(i) = color_saturation(imager);
    CC(i) = CC_evaluation(imageo, imageu, imager);
    NMI(i) = metricsNMI(imageo, imageu, imager);
    Qnice(i) = metricsQncie(imageo, imageu, imager);

end

avg_PSNR = mean(Psnr);
avg_SF = mean(SF);
avg_CS = mean(CS);
avg_CC = mean(CC);
avg_NMI = mean(NMI);
avg_Qnice = mean(Qnice);

data1 = cell(2,7);
title = {'PSNR','SF','CS','CC','NMI','Qnice'};
score = {avg_PSNR,avg_SF,avg_CS,avg_CC,avg_NMI,avg_Qnice};
data1(2,2:end) = score;
Method_name = cellstr(Method_name);
data1(2,1) = Method_name;
data1(1,2:end) = title;
xlswrite(file_name,data1);