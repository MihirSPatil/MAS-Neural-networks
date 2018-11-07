function [plotstructs]= parsespectrallines(Pxlsfile)
%
%struct.xname 'abc'
%struct.xdata [1.0,2.0,...,]
%struct.yname 'my_name'
%struct.ydata [1001,1011,..]
%
%nm,	Rohrspreize PVC hell, (M, id=5921)	Schwamm Trocken, (M, id=5890)	Styrodur, (M, id=5923)	Styrodur verschmutzt, (M, id=5924)	Gummihandschuh Tegera 7380 Armschutz Wolle Stoff, (M, id=5161)	Rohrspreize PVC grau, (M, id=5922)	Gummihandschuh Texxor 2162 Armschutz Stoff, (M, id=5169)	Silikon unverarbeitet, (M, id=6008)	Silikonschlauch unvulkanisiert, (M, id=6009)	Lederhandschuh Artelli Goliath Mittelfinger Stoff, (M, id=5146)
%400	25,09628776	6,558792675	24,25696136	9,87178312	1,675865815	17,7325877	3,148509988	0,748155035	0,993506988	4,60161734
%410	27,91216286	7,387703553	24,92361056	10,40036426	1,718933747	19,29592533	2,653521596	0,739636225	0,989049879	4,614965817

[num,txt] = xlsread(Pxlsfile);
xaxisname1=txt(1);
xdata1=num(:,1)';
for k=1:length(txt)-1
    curstruct= struct(...
                    'xname',xaxisname1,...
                    'xdata',xdata1,...
                    'yname',txt(k+1),...
                    'ydata',num(:,k+1)');
    plotstructs(k)=curstruct;
end
return;
