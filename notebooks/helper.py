import pandas as pd
import os
import re

def get_df_dict(path):
    """Toma un directorio y devuelve un tupla(dic, list), donde el
    diccionario es el hillclimbing de cada grilla y la lista son los
    índices del diccionario ordenado """

    cells_df = {}
    for r, d, f in os.walk(path):
        for file in f:
            match = re.search(".*k:([^_]*).*a:([^_]*)", file)
            k=int(match.group(1))
            alfa=int(match.group(2))
            cells_df[(alfa, k)] = pd.read_csv(path + file)

    sk = list(cells_df.keys())
    sk.sort()

    return cells_df, sk

import numpy as np

def to_numpy(data, rows, cols, d, sk):
    """ transforma el diccionario de busquedas y el índice en una
    matriz numpy lista para plottear """
    grid=[]
    for i in range(0,rows):
        row = []
        for j in range(0,cols):
            row.append(max(d[sk[i*cols + j]][data]))
        grid.append(row)

    return np.array(grid)


from heatmap import *

def fucking_plot_it(np_data, first_y, last_y, step_y, first_x, last_x, step_x):
    fig, ax = plt.subplots(figsize=(24,9))

    ax.set_xlabel('Vecinos')
    ax.set_ylabel('Componentes Principales')

    im, cbar = heatmap(  np_data
                       , [ str(a) for a in range(first_y,last_y+step_y,step_y) ]
                       , [ str(k) for k in range(first_x, last_x+step_x, step_x) ]
                       , ax=ax, cbarlabel='Accuracy')
    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()
    plt.show()

def unique(col):
    ls = []
    for e in col:
        if not e in ls:
            ls.append(e)
    return ls
