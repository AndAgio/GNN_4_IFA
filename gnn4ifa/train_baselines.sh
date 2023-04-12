# List of logs and who should be notified of issues
topos=("small" "dfn" "large")
baselines=("svm" "tree" "forest" "mlp" "bayes")
modes=("avg" "cat")

# Look for signs of trouble in each log
for t in ${!topos[@]};
do
  for b in ${!baselines[@]};
  do
    for m in ${!modes[@]};
    do
      if [ "${topos[$t]}" = "small" ] || [ "${topos[$t]}" = "dfn" ]
      then
        python train_base.py --model="${baselines[$b]}" --train_topology="${topos[$t]}" --data_mode="${modes[$m]}" --train_sims 1 --val_sims 2 3 4 5 --test_sims 2 3 4 5
      else
        python train_base.py --model="${baselines[$b]}" --train_topology="${topos[$t]}" --data_mode="${modes[$m]}" --train_sims 1 --val_sims 2 --test_sims 2
      fi
      done
  done
done