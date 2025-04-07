import yfinance as yf
import numpy as np
import seaborn as sns

# Lendo os dados da bolsa de 2018 até hoje
df = yf.download(["^BVSP"], start="2018-01-01")

# Removendo as colunas duplas e removendo as datas dos índices
df.columns = [col[0] for col in df.columns]
df = df.reset_index()

# Colocando o preço de fechamento na variável s
s = df["Close"]

# Calculando os log-price-returns
r = np.log(s / s.shift(1))

# Calculando o módulo dos log-price-returns
abs_r = np.abs(r)

# Calculando a volatilidade instantânea
v = np.sqrt(np.square(r) - np.square(np.mean(r)))

# Plotando os gráficos para comparação
sns.lineplot(y=s, x=df.index)
sns.lineplot(y=r, x=df.index)
sns.lineplot(y=abs_r, x=df.index)
sns.lineplot(y=v, x=df.index)



