#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing \
              --k-step 5 --alpha-step 25 \
              --grid-k {1..100..10} --grid-alpha {500..50..50} \
              --use-dense-override \
              --out-history big \
              -imp $1
