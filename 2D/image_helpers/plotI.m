function  plotI( I )
%PLOTI �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
len=length(I);
for i=1:len
    subplot(1,len,i),imshow(I{i});

end

