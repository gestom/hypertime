rm test_data_*
rm test_times_*
cat training_data.txt|cut -f 1 -d ' ' >test_times_0.txt
cat training_data.txt|cut -f 2 -d ' ' >test_data_0.txt

for i in $(seq 1 1);do cat test.min|sed -n /$((1457913600+$i*24*60*60*7))/,+10080p |cut -f 2 -d ' ' >test_data_$i.txt;done
for i in $(seq 1 1);do cat test.min|sed -n /$((1457913600+$i*24*60*60*7))/,+10080p |cut -f 1 -d ' ' >test_times_$i.txt;done

e=1;for i in $(seq 2 9);do e=$(($e+1));cat test.min|sed -n /$((1457913600+$i*24*60*60*7-3600))/,+10080p |cut -f 2 -d ' ' >test_data_$e.txt;done
e=1;for i in $(seq 2 9);do e=$(($e+1));cat test.min|sed -n /$((1457913600+$i*24*60*60*7))/,+10080p |cut -f 1 -d ' ' >test_times_$e.txt;done

#e=1;for i in 4 8;do e=$(($e+1));cat test.min|sed -n /$((1457913600+$i*24*60*60*7-3600))/,+10080p |cut -f 2 -d ' ' >test_data_$e.txt;done
#e=1;for i in 4 8;do e=$(($e+1));cat test.min|sed -n /$((1457913600+$i*24*60*60*7))/,+10080p |cut -f 1 -d ' ' >test_times_$e.txt;done
