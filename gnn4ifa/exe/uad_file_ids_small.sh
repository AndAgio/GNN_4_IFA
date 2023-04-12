


python train.py --differential=1 --model="anomaly_gin_2x2x100" --percentile=0.985 --train_scenario="existing" --train_topology="small" --epochs=10 --masking=1
python test.py --differential=1 --model="anomaly_gin_2x2x100" --percentile=0.985 --train_scenario="existing" --train_topology="small" --test_scenario="existing" --test_topology="small" --masking=1