# List of logs and who should be notified of issues
sumduration=0
counter=0

# Look for signs of trouble in each log
for d in {1.0,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.5};
do
  start=$SECONDS
  python train.py --differential=1 --model="anomaly_gin_2x2x100" --percentile=0.985 --train_scenario="existing" --train_topology="small" --epochs=10 --masking=1 --train_freq=$d --split_mode="percentage"
  python test.py --differential=1 --model="anomaly_gin_2x2x100" --percentile=0.985 --train_scenario="existing" --train_topology="small" --test_scenario="existing" --test_topology="small" --masking=1 --train_freq=$d --split_mode="percentage"
  duration=$(( SECONDS - start ))
  sumduration=$((sumduration+duration))
  counter=$((counter+1))
done
avgduration=$(echo "$sumduration/$counter" | bc -l)
echo avgduration: ${avgduration}
avgduration=$( echo "($avgduration+0.5)/1" | bc )
days=$(echo "$avgduration/86400" | bc -l)
hours=$(echo "$avgduration%86400/3600" | bc -l)
minutes=$(echo "$avgduration%3600/60" | bc -l)
seconds=$(echo "$avgduration%60" | bc -l)
echo Run in ${days} d : ${hours} h : ${minutes} m : ${seconds} s on average