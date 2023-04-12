# List of logs and who should be notified of issues
topos=("small" "dfn")
convs=("gcn" "cheb" "gin" "tag" "sg")
percentiles=(0.90 0.905 0.91 0.915 0.92 0.925 0.93 0.935 0.94 0.945 0.95 0.955 0.96 0.965 0.97 0.975 0.98 0.985 0.99 0.995 1)

sumduration=0
counter=0
# Look for signs of trouble in each log
for i in ${!topos[@]};
do
  for j in ${!convs[@]};
  do
    for k in ${!percentiles[@]};
    do
      start=$SECONDS
      python train.py --differential=1 --model="anomaly_${convs[$j]}_2x2x100" --percentile=${percentiles[$k]} --train_scenario="existing" --train_topology="${topos[$i]}" --epochs=10 --masking=1
      duration=$(( SECONDS - start ))
      sumduration=$((sumduration+duration))
      counter=$((counter+1))
    done
  done
done
avgduration=$(echo "$sumduration/$counter" | bc -l)
echo avgduration: ${avgduration}
avgduration=$( echo "($avgduration+0.5)/1" | bc )
days=$(echo "$avgduration/86400" | bc -l)
hours=$(echo "$avgduration%86400/3600" | bc -l)
minutes=$(echo "$avgduration%3600/60" | bc -l)
seconds=$(echo "$avgduration%60" | bc -l)
echo Run in ${days} d : ${hours} h : ${minutes} m : ${seconds} s on average