# List of logs and who should be notified of issues
test_topos=("small" "dfn" "large")
baselines=("poseidon" "coordination" "congestion_aware" "chokifa" "cooperative_filter")

for e in ${!test_topos[@]};
do
  for b in ${!baselines[@]};
  do
    python test_non_ml_base.py --model="${baselines[$b]}" --test_topology="${test_topos[$e]}"
  done
done