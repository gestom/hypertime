set terminal fig color 
set xlabel 'Time [weeks]' offset 0.0,0.2
set ylabel 'Mean squared error [-]' offset 1.2,0.0
set size 0.65,0.75
set title 'Prediction error rate of individiual models'
set key top horizontal 
#set xtics ("Train." 0, "Week 1" 1, "Week 2" 2, "Week 3" 3);
set style fill transparent solid 1.0 noborder;
set boxwidth XXX relative
plot [-0.5:9.5] [0:13]\
