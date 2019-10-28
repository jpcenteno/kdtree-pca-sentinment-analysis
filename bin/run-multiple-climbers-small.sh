#!/bin/bash

# Toma un argumento: la implementación € {sentiment, sklearn}

./searcher.py --algorithm hill-climbing --like-classify \
              --k-step 10 --alpha-step 10 \
              --grid-k {1500..1800..25} --grid-alpha {700..200..50} \
              --use-dense-override \
              --out-history small \
              -imp $1
