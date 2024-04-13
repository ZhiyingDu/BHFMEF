function [cs] = color_saturation(img)
% 计算图像的颜色饱和度
lab = rgb2lab(img); % 将图像转换为 Lab 色彩空间
std_a = std2(lab(:,:,2));
std_b = std2(lab(:,:,3));
L_mean = mean2(lab(:,:,1));
k = 1 / (size(img,1) * size(img,2) - 1);
cs = sqrt(k * sum(sum((lab(:,:,2) - std_a).^2 + (lab(:,:,3) - std_b).^2))) / L_mean;
end