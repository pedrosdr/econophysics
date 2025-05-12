import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
from scipy.special import gamma
import cmath

# ========== PARTE 1: Obter dados do Nasdaq ==========
# Download Nasdaq data from Yahoo Finance
nasdaq = yf.download('^IXIC', start='1997-01-01', end='2000-12-31')
prices = nasdaq['Close'].values
dates = nasdaq.index

# Apply natural logarithm to prices
log_prices = np.log(prices)

# Convert dates to numeric (days since first date)
time_delta = dates - dates[0]
days = time_delta.days.values

# Create Nasdaq plot data
normalized_time_series = np.column_stack((days, log_prices))

# ========== PARTE 2: Modelo de Sornette ==========
def egen(λ, k, x):
    if 1 + λ * x > 0:
        return (1 + λ * x) ** (k / λ)
    else:
        return abs(1 + λ * x) ** (k / λ) * cmath.exp(1j * np.pi * k / λ)

def cosλ(λ, k, x):
    return (egen(λ, k, x)**1j + egen(λ, -k, x)**(-1j)) / 2

def fλ(λ, m, ω, C1, x):
    return egen(λ, m, x) * (1 + C1 * cosλ(λ, ω, x))

# Create model data
x_values = np.linspace(0, 2000, 2000)
model_values = [np.real(fλ(0.5, 0.2, 2.5, 0.3, x)) for x in x_values]

# ========== PARTE 3: Sobrepor os gráficos ==========
plt.figure(figsize=(12, 6))

# Plot Nasdaq data
plt.plot(normalized_time_series[:, 0], normalized_time_series[:, 1], 
         'b-', linewidth=1.5, label='Log Nasdaq')

# Plot Sornette model
plt.plot(x_values, model_values, 'r--', linewidth=1.5, 
         label='Modelo Sornette')

plt.xlabel('Dias desde 01/1997')
plt.ylabel('Valor Normalizado')
plt.title('Comparação: Log Nasdaq vs Modelo de Sornette (1997-2000)')
plt.grid(True)
plt.legend()
plt.show()