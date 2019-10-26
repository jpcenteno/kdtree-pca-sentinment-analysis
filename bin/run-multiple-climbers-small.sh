#!/bin/bash

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 20 --alpha-step 2 \
              --grid-k {100..1000..100} --grid-alpha {90..10..10} \
              --data-set-cut 0.1 \
              -imp $1
