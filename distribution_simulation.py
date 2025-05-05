import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns


def levy(n:int, precision:int=999999) -> np.ndarray:
    """
    Gera uma amostra aleatória a partir da distribuição de Levy
    """
    probs = sp.stats.levy.pdf(np.linspace(0.0, 1000.0, precision))
    probs = probs / probs.sum()
    return np.random.choice(
        np.linspace(0.0, 1000.0, precision),
        n,
        p=probs

    )


# Define o tamanho da amostra
size = 300

# Levy
x = levy(size)
sns.histplot(x=x, bins=int(size/2))
sns.lineplot(x=range(size), y=x)

# Levy com inversão de sinal aleatória
x = levy(size)
pos = np.random.choice([True, False], size)
x = np.where(pos, x, -x)
sns.histplot(x=x, bins=int(size/2))
sns.lineplot(x=range(size), y=x)

# Normal
x = np.random.randn(size)
sns.histplot(x=x, bins=int(size/2))
sns.lineplot(x=range(size), y=x)

# Cauchy
x = np.random.standard_cauchy(size)
sns.histplot(x=x, bins=int(size/2))
sns.lineplot(x=range(size), y=x)
