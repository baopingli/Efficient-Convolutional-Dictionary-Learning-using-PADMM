function [ p ] = psnr( b,DZ )
%PSNR 此处显示有关此函数的摘要
%   此处显示详细说明
it=size(DZ,1);
x1=b;
x2=DZ;
tmp=norm(x1(:)-x2(:));
p=20*log10(it/tmp);
end

