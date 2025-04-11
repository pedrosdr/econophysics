import pandas as pd
import numpy as np
import yfinance as yf
import plotnine as gg
from datetime import datetime

# Define uma data de Início e Fim
start_dt = datetime(year=2017, month=1, day=1)
end_dt = datetime(year=2025, month=4, day=9)

# Lê a base da taxa selic
selic_df = pd.read_csv("selic.csv", sep=";")

# renomeia as colunas
selic_df.columns = ["date", "selic"]

# Remove o texto do footer na última linha
selic_df = selic_df.iloc[:-1,:]

# Converte a coluna data de string para datetime
selic_df["date"] = pd.to_datetime(selic_df["date"], format=("%d/%m/%Y"))

# Remove os separadores de milhar da selic e converte para float64
selic_df["selic"] = selic_df["selic"].str.replace(",","").astype("float64")

# A taxa selic está em percentual, divide por 100 para obter o valor real
selic_df["selic"] = selic_df["selic"] / 100.0

# Calcula o retorno log-preço da selic e converte para taxa diária
selic_df["rf"] = np.log(selic_df["selic"] + 1.0) / 252.0


# Baixa os dados do Ibovespa usando a API yfinance
bvsp_df = yf.download(["^BVSP"], start=start_dt, end=end_dt)

# As colunas estão em formato MultiIndex, converte para formato normal
bvsp_df.columns = [x[0] for x in bvsp_df.columns]

# Mantém apenas os valores de fechamento e renomeia para 's'
bvsp_df = bvsp_df.loc[:,["Close"]]
bvsp_df = bvsp_df.rename(columns={"Close": "s"})

# Calcula o retorno log-preço (nominal)
bvsp_df["rn"] = np.log(bvsp_df["s"] / bvsp_df["s"].shift(1))

# Calcula a volatilidade nominal (módulo do retorno nominal)
bvsp_df["vn"] = np.abs(bvsp_df["rn"])

# Remove os valores nulos (primeira linha ficou nula)
bvsp_df = bvsp_df.dropna()

# Reseta o index para obter a coluna de datas
bvsp_df = bvsp_df.reset_index()

# Deixa os nomes das colunas minúsculos
bvsp_df.columns = bvsp_df.columns.str.lower()

# Faz um left join do dataframe do IBovespa com o dataframe da taxa Selic
# ligando os dois pela data
df = pd.merge(bvsp_df, selic_df, on="date", how="left")

# Datas que não estavam presentes na tabela da taxa selic:
not_matched_values = df[df.isna().any(axis=1)]

# Remove os valores nulos
df = df.dropna()

# Calcula os retornos reais (prêmio de risco)
df["r"] = df["rn"] - df["rf"]

# Calcula a volatilidade real (módulo do retorno real)
df["v"] = np.abs(df["r"])

# Calcula a volatilidade realizada para uma janela de um mês (22 dias úteis)
df["sd"] = df["r"].shift(1).rolling(22).std()


def chart(data, x, y, title="Title", ylab="y", hline=False, yfmt=None):
    fig = gg.ggplot(data=df) + gg.theme_light() +\
        gg.geom_line(mapping=gg.aes(y=y, x=x), size=0.5, color="#4d58bf") +\
        gg.scale_x_datetime(labels=lambda x: [dt.strftime("%b/%Y") for dt in x])
    
    if hline:
        fig = fig + gg.geom_hline(yintercept=0.0, linetype="dashed", size=1.3)
        
    if yfmt:
        fig = fig + gg.scale_y_continuous(
                labels=lambda y: [yfmt(yi) for yi in y]
        )
    
    fig = fig + gg.ggtitle(title) +\
        gg.labs(
            y=ylab
        ) +\
        gg.theme(
           panel_border=gg.element_blank(),
           panel_grid=gg.element_blank(),
           axis_ticks_major_y=gg.element_blank(),
           axis_text_x=gg.element_text(angle=45),
           axis_title_x=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


start_str = start_dt.strftime("%b/%Y")
end_str = end_dt.strftime("%b/%Y")
chart(
      df, 
      "date", 
      "s", 
      title=f"Fechamento do IBovespa de {start_str} a {end_str}",
      ylab="Preço"
)

chart(
      df, 
      "date", 
      "selic", 
      title=f"Taxa Selic de {start_str} a {end_str}",
      ylab="Taxa Selic (base 252 dias)",
      yfmt=lambda x: f"{x*100:.2f}%"
)

chart(
      df, 
      "date", 
      "rn", 
      title=f"Retorno nominal de {start_str} a {end_str}",
      ylab="Retorno nominal (rn)",
      hline=True
)

chart(
      df, 
      "date", 
      "r", 
      title=f"Retorno real de {start_str} a {end_str}",
      ylab="Retorno real (r)",
      hline=True
)

chart(
      df, 
      "date", 
      "vn", 
      title=f"Volatilidade nominal de {start_str} a {end_str}",
      ylab="Volatilidade nominal |rn|",
      hline=True
)

chart(
      df, 
      "date", 
      "v", 
      title=f"Volatilidade real de {start_str} a {end_str}",
      ylab="Volatilidade real |r|",
      hline=True
)