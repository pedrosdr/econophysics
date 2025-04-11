import yfinance as yf
import numpy as np
import seaborn as sns
import plotnine as gg

# Lendo os dados da bolsa de 2018 até hoje
start_date = ["2007", "01", "01"]
end_date = ["2010", "04", "08"]
df = yf.download(
    ["^BVSP"], 
    start="-".join(start_date), 
    end="-".join(end_date)
)

# Removendo as colunas duplas e removendo as datas dos índices
df.columns = [col[0] for col in df.columns]
df = df.reset_index()

# Selecionando o preço de fechamento
s = df["Close"]

# Calculando o log-price-return
r = np.log(s / s.shift(1))
r = r.dropna()

# Calculando a volatilidade instantânea
v = np.sqrt(np.square(r) - np.square(np.mean(r)))

# Calculando a função de autocorrelação para r e v
Rr = [((r.shift(i) * r).mean() - r.mean()**2) / r.std()**2 for i in range(100)]
Rv = [((v.shift(i) * v).mean() - v.mean()**2) / v.std()**2 for i in range(100)]

# Plotando os gráficos para comparação
sns.lineplot(y=s, x=range(len(s)))
sns.lineplot(y=r, x=range(len(r)))
sns.lineplot(y=v, x=range(len(v)))
sns.lineplot(y=np.log(Rr[1:]), x=np.log(list(range(100))[1:]))
sns.lineplot(y=np.log(Rv), x=np.log(range(100)))

start_date_formatado = "/".join(list(reversed(start_date)))
end_date_formatado = "/".join(list(reversed(end_date)))
gg.ggplot() + \
    gg.geom_line(gg.aes(x=df.Date, y=s)) + \
    gg.scale_x_datetime(date_labels="%d/%m/%Y") + \
    gg.theme_light() + \
    gg.labs(
        y="Fechaamento"
    ) + \
    gg.ggtitle(
        f"Fechamento do IBovespa de {start_date_formatado} a {end_date_formatado}"
    ) + \
    gg.theme(
     panel_border=gg.element_blank(),
     axis_title_x=gg.element_blank(),
     axis_text_x=gg.element_text(angle=45)
    )
