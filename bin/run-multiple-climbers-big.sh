#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 10 --alpha-step 10 \
              --grid-k {100..2000..100} --grid-alpha {1100..500..100} \
              --use-dense-override \
              --out-history big \
              -imp $1
