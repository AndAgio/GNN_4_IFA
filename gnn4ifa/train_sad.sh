# List of logs and who should be notified of issues
topos=("small" "dfn")
convs=("gin")
#convs=("gcn" "cheb" "gin" "tag" "sg")
pools=("mean") # "sum" "max" "s2s" "att")
mus=(1 2 3 4 5 6 7 8)

sumduration=0
counter=0
# Look for signs of trouble in each log
for i in ${!topos[@]};
do
  for j in ${!convs[@]};
  do
    for k in ${!mus[@]};
    do
      for p in ${!pools[@]};
      do
        start=$SECONDS
        python train.py --model="class_${convs[$j]}_${mus[$k]}x100_${pools[$p]}" --train_scenario="existing" --train_topology="${topos[$i]}" --epochs=10 --differential=0 --masking=0 --train_sims 1 2 --val_sims 2 --test_sims 3 4 5
        duration=$(( SECONDS - start ))
        sumduration=$((sumduration+duration))
        counter=$((counter+1))
      done
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