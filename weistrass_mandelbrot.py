import numpy as np
import plotnine as gg


def plot(x, y, title=""):
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
        gg.labs(
            x="t",
            y="W"
        ) +\
        gg.ggtitle(
            title
        ) +\
        gg.theme(
            panel_grid_major=gg.element_blank(),
            panel_grid_minor=gg.element_blank()
        )
        
    return fig


def C(t, gamma, D, Nmax=500):
    n = np.arange(-Nmax, Nmax+1)
    
    gam_n = gamma**n
    gam_n_t = np.outer(gam_n, t)
    top = 1 - np.cos(gam_n_t)
    bottom = np.power(gamma, (2-D)*n)
    
    c = top / np.expand_dims(bottom, 1)
    return c.sum(axis=0)


def W(t, gamma, D, mu, Nmax=500):
    n = np.arange(-Nmax, Nmax+1)
    
    phi_n = mu*n
    gam_n = gamma**n
    gam_n_t = np.outer(gam_n, t)
    top_left = 1-np.exp(1j*gam_n_t)
    top_right = np.exp(1j*phi_n)
    top = top_left * np.expand_dims(top_right, 1)
    bottom = np.power(gamma, (2-D)*n)
    
    c = top / np.expand_dims(bottom, 1)
    return c.sum(axis=0)


x = np.linspace(0.0, 1.0, 100000)
y2a = W(x, gamma=1.5, D=1.0, mu=0.0)
y2b = W(x, gamma=1.5, D=1.2, mu=0.0)
y2c = W(x, gamma=1.5, D=1.5, mu=0.0)
y2d = W(x, gamma=1.5, D=1.8, mu=0.0)
y2e = W(x, gamma=1.5, D=1.99, mu=0.0)

plot(x, y2e.real)
