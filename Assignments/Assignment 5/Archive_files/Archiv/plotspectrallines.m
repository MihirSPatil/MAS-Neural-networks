function plotspectrallines(Pplotstructs)
%function plotspectrallines(Pplotstructs)
%    xname: 'nm'
%    xdata: [1x121 double]
%    yname: 'Schwamm Trocken, (M, id=5890)'
%    ydata: [1x121 double]

ColorVal ='ymcrgbk';
MarkerSymbol = '+o*.xsd^v<>ph';
LineStyle{1} = '-';
LineStyle{2} = '--';
LineStyle{3} = ':';
LineStyle{4} = '_.';
for k=1:length(Pplotstructs)
    mIdx = mod(k,length(MarkerSymbol))+1;
    cIdx = mod(k,length(ColorVal))+1;
    lIdx = mod(k,length(LineStyle))+1;
    doofman = char(LineStyle{lIdx});
 plot(Pplotstructs(k).xdata,...
      Pplotstructs(k).ydata,...
     'Color',ColorVal(cIdx),...
     'Marker',MarkerSymbol(mIdx))
 %,...
 %    'LineStyle',doofman);
 hold on;
end
%legend(Pplotstructs.yname);
return;