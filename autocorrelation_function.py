import yfinance as yf
import numpy as np
import seaborn as sns

# Lendo os dados da bolsa de 2018 até hoje
df = yf.download(["^BVSP"], start="2018-01-01", end="2025-01-01")

# Removendo as colunas duplas e removendo as datas dos índices
df.columns = [col[0] for col in df.columns]
df = df.reset_index()

# Colocando o preço de fechamento na variável s
s = df["Close"]

# Calculando o log-price-return
r = np.log(s / s.shift(1))
r = r.dropna()

# Calculando a volatilidade
v = np.abs(r)

# Calculando a função de autocorrelação para r e v
Rr = [((r.shift(i) * r).mean() - r.mean()**2) / r.std()**2 for i in range(100)]
Rv = [((v.shift(i) * v).mean() - v.mean()**2) / v.std()**2 for i in range(100)]

# Plotando os gráficos para comparação
sns.lineplot(y=s, x=range(len(s)))
sns.lineplot(y=r, x=range(len(r)))
sns.lineplot(y=v, x=range(len(v)))
sns.lineplot(y=Rr, x=range(100))
sns.lineplot(y=np.log(Rv), x=np.log(range(100)))
