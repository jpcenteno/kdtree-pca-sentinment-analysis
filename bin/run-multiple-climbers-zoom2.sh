#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing\
              --k-step 10 --alpha-step 50 \
              --grid-k {50..150..20} --grid-alpha {500..800..100} \
              --use-dense-override \
              --out-history zoom2 \
              -imp $1
