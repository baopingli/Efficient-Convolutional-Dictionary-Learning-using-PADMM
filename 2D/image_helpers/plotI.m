function  plotI( I )
%PLOTI 此处显示有关此函数的摘要
%   此处显示详细说明
len=length(I);
for i=1:len
    subplot(1,len,i),imshow(I{i});

end

