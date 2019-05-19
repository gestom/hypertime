d=$1

function extend_figure
{
	w=$(identify $1 |cut -f 3 -d ' '|cut -f 1 -d x)
	h=$(identify $1 |cut -f 3 -d ' '|cut -f 2 -d x)
	if [ $w -lt 500 ]; then	convert $1 -bordercolor white -border $(((500-$w)/2))x0 $1;fi
	if [ $h -lt 400 ]; then	convert $1 -bordercolor white -border 0x$(((400-$h)/2)) $1;fi
	convert $1 -resize 500x400 $1
	w=$(identify $1 |cut -f 3 -d ' '|cut -f 1 -d x)
	h=$(identify $1 |cut -f 3 -d ' '|cut -f 2 -d x)
	if [ $w -lt 500 ]; then	convert $1 -bordercolor white -border $(((500-$w)/2))x0 $1;fi
	if [ $h -lt 400 ]; then	convert $1 -bordercolor white -border 0x$(((400-$h)/2)) $1;fi
	convert $1 -resize 500x400 $1
}

function create_graph
{
	echo digraph 
	echo { 
	echo node [penwidth="2" fontname=\"palatino bold\"]; 
	echo edge [penwidth="2"]; 
	for m in $(cut -f 1 -d ' ' tmp/models.tmp)
	do	
		e=0
		for n in $(cut -f 1 -d ' ' tmp/models.tmp)
		do
			#echo -ne Comparing $m and $n' ';
			if [ $(paste tmp/$m.txt tmp/$n.txt|tr \\t ' '|cut -f 1,3 -d ' '|./t-test|grep -c higher) == 1 ] 	#change fields 1,3 to 2,4 for absolute testing
			then
				echo \"$(grep $n tmp/best.txt|cut -d ' ' -f 2,4|sed s/' '/_/|sed s/\_0//)\" '->' \"$(grep $m tmp/best.txt|cut -d ' ' -f 2,4|sed s/' '/_/|sed s/\_0//)\" ;
				e=1
			fi
		done
		if [ $e == 0 ]; then echo \"$(grep $m tmp/best.txt|cut -d ' ' -f 2,4|sed s/' '/_/|sed s/\_0//)\";fi
	done
	echo }
}

mkdir tmp
rm tmp/best.txt
grep -v '#' ../src/models/test_models.txt|tr -d '!' >tmp/models.tmp
for m in $(cut -f 1 -d ' ' tmp/models.tmp)
do
	errmin=100
	indmin=0
	for o in $(cat tmp/models.tmp |grep $m|sed  -e 's/\s\+/\ /g'|cut -f 2-100 -d ' ');
	do
		err=$(cat ../results/$d/$m\_$o.txt|awk 'NR==1{a=0}{a=a+$1}END{print a}')				#change field $2 to $1
		sm=$(echo $err $errmin|awk '{a=0}($1 > $2){a=1}{print a}')
		if [ $sm == 0 ];
		then
			errmin=$err
			indmin=$o
		fi
	done
	cat ../results/$d/$m\_$indmin.txt >tmp/$m.txt
	echo Model $m param $indmin has $errmin error.  >>tmp/best.txt
done
cat tmp/best.txt|sort -k 6 >tmp/besta.txt
mv tmp/besta.txt tmp/best.txt

create_graph >a.txt 
create_graph |dot -Tpng >tmp/$d.png
convert tmp/$d.png -trim -bordercolor white tmp/$d.png 
extend_figure tmp/$d.png
cat tmp/best.txt |cut -f 2,4 -d ' '|tr ' ' _|sed s/$/.txt/|sed s/^/..\\/results\\/$d\\//
f=0
n=$((1+$(cat tmp/models.tmp|grep -c .)));
cat draw_summary_skelet.gnu|sed s/XXX/1.0\\/$n/ >tmp/draw_summary.gnu
for i in $(cut -f 2 -d ' ' tmp/best.txt);
do 
	echo \'$(grep $i tmp/best.txt |cut -f 2,4 -d ' '|tr ' ' _|sed s/$/.txt/|sed s/^/..\\/results\\/$d\\//)\' 'using ($0+XXX):($1*100) with boxes title' \'$i\',\\|sed s/XXX/\($f-$n*0.5+1\)\\/$n\.0/ >>tmp/draw_summary.gnu;   #modify for another error measure
	f=$(($f+1))
done
gnuplot tmp/draw_summary.gnu >tmp/graphs.fig
fig2dev -m5 -Lpng tmp/graphs.fig tmp/graphs.png
convert tmp/graphs.png -trim -resize 500x400 tmp/graphsx.png
extend_figure tmp/graphsx.png 
convert -size 900x450 xc:white \
	-draw 'Image src-over 25,50 500,400 'tmp/graphsx.png'' \
	-draw 'Image src-over 525,90 375,300 'tmp/$d.png'' \
	-pointsize 24 \
	-draw 'Text 140,30 "Performance of temporal models for door state prediction"' \
	-pointsize 18 \
	-gravity North \
	-draw 'Text 0,40 "Arrow A->B means that A performs statistically significantly better that B"' tmp/summary.png;
cp tmp/summary.png  ../results/summary.png
cp tmp/summary.png  summary.png
