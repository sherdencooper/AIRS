#!/bin/bash

nohup python -u fidelity_test.py --method concept > fidelity_log/concept.log 2>&1 &
nohup python -u fidelity_test.py --method attention > fidelity_log/attention.log 2>&1 &
nohup python -u fidelity_test.py --method theta > fidelity_log/theta.log 2>&1 &
nohup python -u fidelity_test.py --method saliency > fidelity_log/saliency.log 2>&1 &
nohup python -u fidelity_test.py --method smooth > fidelity_log/smooth.log 2>&1 &
nohup python -u fidelity_test.py --method integrated > fidelity_log/integrated.log 2>&1 &
nohup python -u fidelity_test.py --method random > fidelity_log/random.log 2>&1 &