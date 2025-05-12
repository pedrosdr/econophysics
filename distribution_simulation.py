import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import plotnine as gg


def distplot(x, min_y=None, max_y=None, title=None):
    plot = gg.ggplot() + gg.theme_light() +\
        gg.geom_line(
            mapping = gg.aes(
                x=np.arange(len(x)),
                y=x
            )    
        ) +\
        gg.labs(
            y="Variável X",
            x="Tempo"
        ) +\
        gg.theme(
            panel_grid=gg.element_blank(),
            plot_title=gg.element_text(hjust=0.5)
        )
    
    if min_y is not None and max_y is not None:
        plot = plot +\
            gg.coord_cartesian(ylim=[min_y, max_y])
            
    if title is not None:
        plot = plot +\
            gg.ggtitle(title)
            
    return plot


# Define o tamanho da amostra
size = 1000

# Levy
x = sp.stats.levy.rvs(size = size)
distplot(x, title="Distribuição Levy Padrão")
distplot(x, min_y=0, max_y=30, title="Distribuição Levy Padrão (Com zoom)")

# Normal
x = np.random.randn(size)
distplot(x, title="Distribuição Normal Padrão")

# Cauchy
x = sp.stats.cauchy.rvs(size=size)
distplot(x, title="Distribuição Cauchy (Com zoom)")
distplot(x, min_y=-4, max_y=4, title="Distribuição Cauchy (Com zoom)")
