#!/usr/env bash

python random_attack.py --data alpha --budget 100 --frac 0.2 --alg fraudar &
python random_attack.py --data alpha --budget 100 --frac 0.2 --alg rsd &
python random_attack.py --data alpha --budget 100 --frac 0.2 --alg rev2 &

python random_attack.py --data otc --budget 100 --frac 0.2 --alg fraudar &
python random_attack.py --data otc --budget 100 --frac 0.2 --alg rsd &
python random_attack.py --data otc --budget 100 --frac 0.2 --alg rev2 &

wait;

python random_attack.py --data amazon --budget 100 --frac 0.2 --alg fraudar &
python random_attack.py --data amazon --budget 100 --frac 0.2 --alg rsd &
python random_attack.py --data amazon --budget 100 --frac 0.2 --alg rev2 &

python random_attack.py --data epinions --budget 100 --frac 0.2 --alg fraudar &
python random_attack.py --data epinions --budget 100 --frac 0.2 --alg rsd &
python random_attack.py --data epinions --budget 100 --frac 0.2 --alg rev2 &

wait;