function [ p ] = psnr( b,DZ )
%PSNR �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
it=size(DZ,1);
x1=b;
x2=DZ;
tmp=norm(x1(:)-x2(:));
p=20*log10(it/tmp);
end

