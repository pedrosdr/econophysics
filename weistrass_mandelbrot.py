import numpy as np
import plotnine as gg


def plot(x, y):
    fig = gg.ggplot() + gg.theme_light() +\
        gg.geom_line(
            mapping=gg.aes(
                x=x,
                y=y
            )    
        ) +\
        gg.scale_y_continuous(
            labels=lambda x: [f"{xi:.2f}" for xi in x]
        ) +\
        gg.theme(
            panel_grid_major=gg.element_blank(),
            panel_grid_minor=gg.element_blank()
        )
        
    return fig


def C(t, gamma, D, Nmax=50):
    n = np.arange(-Nmax, Nmax+1)
    
    gam_n = gamma**n
    gam_n_t = np.outer(gam_n, t)
    top = 1 - np.cos(gam_n_t)
    bottom = np.power(gamma, (2-D)*n)
    
    c = top / np.expand_dims(bottom, 1)
    return c.sum(axis=0)


x = np.linspace(0.0, 1.0, 100000)
y = C(x, 1.5, 1.2)

plot(x, y)


import matplotlib.pyplot as plt
def C_fast(t, gamma, D, Nmax=1000):
    n     = np.arange(-Nmax, Nmax+1)               # shape (2 Nmax+1,)
    gam_n = gamma**n                               
    denom = gamma**((2.0-D)*n)                    

    # np.outer(gam_n, t) tem shape (2Nmax+1, len(t))
    top   = 1.0 - np.cos(np.outer(gam_n, t))      
    termos= top / denom[:, None]          

    # soma sobre o eixo dos n → vetor de tamanho len(t)
    return termos.sum(axis=0)

y_fast = C_fast(x, gamma=1.5, D=1.5, Nmax=1000)
y_fast = C(x, gamma=1.5, D=1.5, size=1000)
plt.plot(x, y_fast)
plt.show()

plot(x, y_fast)

np.outer(np.arange(0, 100), np.arange(0,100))

a = np.arange(100)
a[:None]

import numpy as np
x = np.linspace(0,1,10000)
y1 = C(x, 1.5, 1.2, size=1000)
y2 = C_fast(x, 1.5, 1.2, Nmax=1000)
print(np.allclose(y1, y2, rtol=1e-6, atol=1e-8)) 


gamma = 1.5
D = 1.5
t = np.linspace(0, 1, 5)
n = np.arange(-3, 3+1)
gam_n = gamma**n
gam_n_t = np.outer(gam_n, t)

top = 1.0 - np.cos(gam_n_t)
bottom = gamma**((2-D)*n)
bottom = np.expand_dims(bottom, 1)

a = top / bottom
a.sum(axis=0)
