# List of logs and who should be notified of issues
train_topos=("small" "dfn" "large")
test_topos=("small" "dfn" "large")
baselines=("svm" "tree" "forest" "mlp" "bayes")
modes=("avg" "cat")

# Look for signs of trouble in each log

for b in ${!baselines[@]};
do
  for m in ${!modes[@]};
  do
    for r in ${!train_topos[@]};
    do
      for e in ${!test_topos[@]};
      do
        if [ "${test_topos[$e]}" = "small" ] || [ "${test_topos[$e]}" = "dfn" ]
        then
          python test_base.py --model="${baselines[$b]}" --data_mode="${modes[$m]}" --train_topology="${train_topos[$r]}" --train_sims 1 --test_topology="${test_topos[$e]}" --test_sims 2 3 4 5
        else
          python test_base.py --model="${baselines[$b]}" --data_mode="${modes[$m]}" --train_topology="${train_topos[$r]}" --train_sims 1 --test_topology="${test_topos[$e]}" --test_sims 2
        fi
      done
    done
  done
done