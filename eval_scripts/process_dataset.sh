dataset=$1
grep -v '!' ../src/models/test_models.txt|grep -v '#' >models.tmp
mkdir ../results/$dataset/
for m in $(cut -f 1 -d ' ' models.tmp)
do	
rm ../results/$dataset/$m*
for i in $(cat models.tmp |grep $m|sed  -e 's/\s\+/\ /g'|cut -f 2-100 -d ' ');
do
rm predictions.txt

d=0
echo ../bin/fremen ../data/$dataset/training_data.txt ../data/$dataset/test_times_$d.txt $m $i
../bin/fremen ../data/$dataset/training_data.txt ../data/$dataset/test_times_$d.txt $m $i
e=$(paste predictions.txt ../data/$dataset/test_data_$d.txt |awk '{a=$1-$2;c=($1>0.5)-$2;b+=a*a;d+=c*c;i=i+1;}END{print b/i,d/i}')
echo $e >>../results/$dataset/$m\_$i.txt

echo Model $m, parameter $i
a=$(ls ../data/$dataset/test_data_*.txt|wc -l)
for d in $(seq 1 $(($a-1)))
do 
echo ../bin/fremen ../data/$dataset/training_data.txt ../data/$dataset/test_times_$d.txt $m model 
../bin/fremen ../data/$dataset/training_data.txt ../data/$dataset/test_times_$d.txt $m model 
e=$(paste predictions.txt ../data/$dataset/test_data_$d.txt |awk '{a=$1-$2;c=($1>0.5)-$2;b+=a*a;d+=c*c;i=i+1;}END{print b/i,d/i}')
echo $e >>../results/$dataset/$m\_$i.txt
echo Model $m, parameter $i
done
done
done
