#!/bin/bash

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 10 --alpha-step 10 \
              --grid-k {100..1000..100} --grid-alpha {1100..500..100} \
              -imp $1
