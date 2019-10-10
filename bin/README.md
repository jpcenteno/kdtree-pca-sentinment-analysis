
### searcher.py

busqueda local de hiperparámetros

*ejemplos:*

``` bash
python3 searcher.py --algorithm hill_climbing sentiment
```

lo mismo pero sin usar PCA

``` bash
./searcher.py --algorithm hill_climbing --not-use-pca sentiment
```

... and so on ...


algunos para considerar

./searcher.py --algorithm grid-beam -k 400 --alpha 100 --k-step 40 --alpha-step 10 --beam-size 10 --divition-scale 80 sentiment
