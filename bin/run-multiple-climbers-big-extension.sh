#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing \
              --k-step 6 --alpha-step 25 \
              --grid-k {40..100..12} --grid-alpha {400..50..50} \
              --use-dense-override \
              --out-history big_ext \
              -imp $1
