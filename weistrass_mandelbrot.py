import numpy as np
import plotnine as gg
from scipy.optimize import least_squares
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


#-----------------------------------------------------------------------
# Utility plotting function using plotnine
#-----------------------------------------------------------------------
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
            panel_grid_minor=gg.element_blank(),
            plot_title=gg.element_text(hjust=0.5)
        )
        
    return fig


#-----------------------------------------------------------------------
# Compute the Weierstrass–Mandelbrot function W(t)
#-----------------------------------------------------------------------
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


#-----------------------------------------------------------------------
# Generate input time vector
#-----------------------------------------------------------------------
x = np.linspace(0.0, 1.0, 100000)

# Compute real part of W for various fractal dimensions D at mu=0
#-----------------------------------------------------------------------
y1a = W(x, gamma=1.5, D=1.0, mu=0.0)
y1b = W(x, gamma=1.5, D=1.2, mu=0.0)
y1c = W(x, gamma=1.5, D=1.5, mu=0.0)
y1d = W(x, gamma=1.5, D=1.8, mu=0.0)
y1e = W(x, gamma=1.5, D=1.99, mu=0.0)

# Plot C(t) for different dimensions
print(plot(x, y1a.real, title="C(t): γ=1.5, D=1.0"))
print(plot(x, y1b.real, title="C(t): γ=1.5, D=1.2"))
print(plot(x, y1c.real, title="C(t): γ=1.5, D=1.5"))
print(plot(x, y1d.real, title="C(t): γ=1.5, D=1.8"))
print(plot(x, y1e.real, title="C(t): γ=1.5, D=1.99"))


# Compute imaginary part (scaled) for mu=pi to get A(t)
#-----------------------------------------------------------------------
y2a = W(x, gamma=1.5, D=1.2, mu=np.pi)
y2b = W(x, gamma=1.5, D=1.5, mu=np.pi)
y2c = W(x, gamma=1.5, D=1.8, mu=np.pi)
y2d = W(x, gamma=1.5, D=1.99, mu=np.pi)

# Plot A(t) = -Imag[W(t)] for different dimensions
print(plot(x, -y2a.imag, title="A(t): γ=1.5, D=1.2"))
print(plot(x, -y2b.imag, title="A(t): γ=1.5, D=1.5"))
print(plot(x, -y2c.imag, title="A(t): γ=1.5, D=1.8"))
print(plot(x, -y2d.imag, title="A(t): γ=1.5, D=1.99"))

# Zoom into a small range for comparison
#-----------------------------------------------------------------------
mask = (x >= 0.3) & (x <=0.31)
_x = x[mask]
_y1c = y1c[mask]
_y1d = y1d[mask]
_y2b = y2b[mask]
_y2c = y2c[mask]

print(plot(_x, _y1c.real, title="C(t): γ=1.5, D=1.5, 0.30 <= t <= 0.31"))
print(plot(_x, _y1d.real, title="C(t): γ=1.5, D=1.8, 0.30 <= t <= 0.31"))
print(plot(_x, -_y2b.imag, title="A(t): γ=1.5, D=1.5, 0.30 <= t <= 0.31"))
print(plot(_x, -_y2c.imag, title="A(t): γ=1.5, D=1.8, 0.30 <= t <= 0.31"))


#-----------------------------------------------------------------------
# Load and normalize historical price data (BVSP)
#-----------------------------------------------------------------------
start_dt = datetime(2000,11,1)
end_dt = datetime(2025,1,1)

# Read CSV, filter dates, set index
df = (pd.read_csv("bvsp.csv", parse_dates=["Date"])
             .query("Date > @start_dt and Date < @end_dt")
             .set_index("Date"))

# Extract closing prices and scale to [0,1]
y = df["Close"].to_numpy()
scaler = MinMaxScaler()
y = scaler.fit_transform(y[:,None]).reshape(-1)

# Create normalized time axis
x = np.linspace(0.0, 1.0, len(y))

#-----------------------------------------------------------------------
# Nonlinear least-squares fit of fractal model to price series
#-----------------------------------------------------------------------
def residuals(params, x, y):
    a, gamma, D, mu = params
    return y - (a + W(x, gamma, D, mu).real)

# bounds = (
#     [-np.inf, 1.5,    1.0,    0.0],
#     [ np.inf, 2.0, np.inf, np.inf]    
# )

# res = least_squares(
#     residuals,
#     x0=[0.0, 1.5, 1.5, 0.0],
#     args=[x, y],
#     bounds=bounds,
#     max_nfev=100,
#     verbose=2
# )
# params = res.x
params = [-7.61798207e-03,  1.50000000e+00,  1.00678807e+00,  5.76455997e-05]

# predicted series from fitted model
a, gamma, D, mu = params
ypred = a + W(x, gamma, D, mu).real

# Plot observed vs fitted series
sns.lineplot(x=x, y=y)
sns.lineplot(x=x, y=ypred)
