#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing \
              --k-step 5 --alpha-step 5 \
              --grid-k {1800..2100..25} --grid-alpha {500..1000..75} \
              --use-dense-override \
              --out-history zoom1 \
              -imp $1
