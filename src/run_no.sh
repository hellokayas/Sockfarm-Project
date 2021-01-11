#!/usr/env bash

python no_attack.py --data alpha --alg fraudar &
python no_attack.py --data alpha --alg rsd &
python no_attack.py --data alpha --alg rev2 &

python no_attack.py --data otc --alg fraudar &
python no_attack.py --data otc --alg rsd &
python no_attack.py --data otc --alg rev2 &

wait;

python no_attack.py --data amazon --alg fraudar &
python no_attack.py --data amazon --alg rsd &
python no_attack.py --data amazon --alg rev2 &

python no_attack.py --data epinions --alg fraudar &
python no_attack.py --data epinions --alg rsd &
python no_attack.py --data epinions --alg rev2 &

wait;
