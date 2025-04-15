import numpy as np
import plotnine as gg


def rnorm(n:int, m:float, sd:float):
    x = np.linspace(m-5.0*sd, m+5.0*sd, num=9999999)
    p = np.exp(
        -np.square(x-m) / (2.0*np.square(sd))
    ) / (sd*np.sqrt(2.0*np.pi))
    p = p / np.sum(p)
    return np.random.choice(a=x, p=p, size=n)


def rcl(n:int, m:float, gamma:float):
    x = np.linspace(m-20.0*gamma, m+20.0*gamma, num=9999999)
    p = (1.0/np.pi) * (gamma / (np.square(x-m) + np.square(gamma)))
    p = p / np.sum(p)
    return np.random.choice(a=x, p=p, size=n)
    

def print_dist(func, bins:int, *args, **kwargs):
    fig = gg.ggplot() + gg.theme_light() +\
        gg.geom_histogram(
            mapping=gg.aes(
                x=func(*args, **kwargs)
            ),
            bins=bins
        )
    return fig

print_dist(rnorm, bins=400, n=300000, m=3, sd=1)
print_dist(rcl, bins=400, n=300000, m=3, gamma=1)

