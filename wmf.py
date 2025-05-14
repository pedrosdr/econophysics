import numpy as np
import seaborn as sns
import plotly.graph_objects as go


@np.vectorize
def weierstrass(t, a=1.0, b=1.0, nmax=200):
    n = np.arange(nmax)
    w = (a**n)*np.cos((b**n)*np.pi*t)
    w = w.sum()
    return w

t = np.linspace(0, 10, 100)
w = weierstrass(t, a=0.5, b=6.2)
sns.lineplot(x=t, y=w)


@np.vectorize
def wmf(t, gamma=1.5, D=1.5, phi=0.0, nmax=200):
    n_lower = np.floor(-(nmax / 2.0))
    n_upper = np.ceil(nmax / 2.0)
    n = np.arange(n_lower, n_upper)
    
    up = (1.0 - np.exp(1j*(gamma**n)*t))*np.exp(1j*phi)
    down = gamma**((2.0-D)*n)
    
    return (up/down).sum()


@np.vectorize
def re_wmf(t, gamma=1.5, D=1.5, nmax=200):
    n_lower = np.floor(-(nmax / 2.0))
    n_upper = np.ceil(nmax / 2.0)
    n = np.arange(n_lower, n_upper)
    
    up = 1 - np.cos((gamma**n)*t)
    down = gamma**((2.0-D)*n)
    
    return (up/down).sum()


@np.vectorize
def im_wmf(t, gamma=1.5, D=1.5, nmax=200):
    n_lower = np.floor(-(nmax / 2.0))
    n_upper = np.ceil(nmax / 2.0)
    n = np.arange(n_lower, n_upper)
    
    up = ((-1)**n)*np.sin((gamma**n)*t)
    down = gamma**((2.0-D)*n)
    
    return (up/down).sum()

    
# Plotting WMF (phi = 0: re_wmf)
t = np.linspace(0, 1, 100)
w = wmf(t, gamma=2.2, D=1.5, phi=0).real
sns.lineplot(x=t, y=w)

t = np.linspace(0, 1, 100)
w = wmf(t, gamma=3.2, D=1.5, phi=0).imag
sns.lineplot(x=t, y=w)

t = np.linspace(0, 1, 100)
w = re_wmf(t, gamma=3.2, D=1.5)
sns.lineplot(x=t, y=w)

# Plotting WMF (phi = n*pi: im_wmf)
t = np.linspace(0, 1, 100)
w = im_wmf(t, gamma=3.2, D=1.5)
sns.lineplot(x=t, y=w)


# Simulating a 3D graph
D = np.linspace(1.5, 2.0, 100)
gamma = np.linspace(1.6, 2.0, 100)
t = np.linspace(0.0, 1.0, 100)

# Creating a surface with fixed D
t_m, gamma_m = np.meshgrid(t, gamma)

D0 = 1.5
z = wmf(t=t_m, gamma=gamma_m, D=D0).real

fig = go.Figure(
    data=go.Surface(
        x=t,
        y=gamma,
        z=z,
        colorscale="Viridis",
        showscale=True
    )
).update_layout(
    title=f"WMF surface at D={D0}",
    scene=dict(
        xaxis_title="t",
        yaxis_title="gamma",
        zaxis_title="WMF"
    ),
    width=700,
    height=600,
    margin=dict(l=60, r=60, b=60, t=80)
)
    
fig.show(renderer="browser")


# Creating a surface with fixed gamma
t_m, D_m = np.meshgrid(t, D)

gamma0 = 1.0
z = wmf(t=t_m, gamma=gamma0, D=D_m).real

fig = go.Figure(
    data=go.Surface(
        x=t,
        y=D,
        z=z,
        colorscale="Viridis",
        showscale=True
    )
).update_layout(
    title=f"WMF surface at gamma={gamma0}",
    scene=dict(
        xaxis_title="t",
        yaxis_title="D",
        zaxis_title="WMF"
    ),
    width=700,
    height=600,
    margin=dict(l=60, r=60, b=60, t=80)
)
    
fig.show(renderer="browser")


# Creating a surface with fixed t
gamma_m, D_m = np.meshgrid(gamma, D)

t0 = 0.5
z = wmf(t=t0, gamma=gamma_m, D=D_m).real

fig = go.Figure(
    data=go.Surface(
        x=gamma_m,
        y=D_m,
        z=z,
        colorscale="Viridis",
        showscale=True
    )
).update_layout(
    title=f"WMF surface at t={t0}",
    scene=dict(
        xaxis_title="gamma",
        yaxis_title="D",
        zaxis_title="WMF"
    ),
    width=700,
    height=600,
    margin=dict(l=60, r=60, b=60, t=80)
)
    
fig.show(renderer="browser")

