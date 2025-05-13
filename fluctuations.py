import pandas as pd
import numpy as np
import yfinance as yf
import plotnine as gg
from datetime import datetime, timedelta
from typing import List
from scipy.stats import kstest
import scipy as sp
import seaborn as sns

# Define uma data de Início e Fim
start_dt = datetime(year=1980, month=1, day=1)
end_dt = datetime(year=2025, month=5, day=11)

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


# # Baixa os dados do Ibovespa usando a API yfinance
# bvsp_df = yf.download(["^BVSP"], start=start_dt, end=end_dt)
  
# # As colunas estão em formato MultiIndex, converte para formato normal
# bvsp_df.columns = [x[0] for x in bvsp_df.columns]
# bvsp_df.to_csv("bvsp.csv")

# Lê o arquivo CSV do BVSP
bvsp_df = pd.read_csv("bvsp.csv")
bvsp_df["Date"] = pd.to_datetime(bvsp_df["Date"])
bvsp_df = bvsp_df[bvsp_df["Date"] > start_dt]
bvsp_df = bvsp_df[bvsp_df["Date"] < end_dt]
bvsp_df = bvsp_df.set_index("Date")

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


# Calcula a volatilidade instantânea (alguns valores podem ficar NA)
vi_square = np.square(df["r"]) - np.square(df["r"].mean()) 
vi_square[vi_square < 0] # Valores negativos

vi = np.sqrt(vi_square)
vi[vi.isna()] # Os valores negativos ficaram NA
df["vi"] = vi

# Define uma função para plotar os gráficos usanso ggplot
def chart(
        data: pd.DataFrame, 
        x: str, 
        y: str, 
        title: str = "Title", 
        ylab: str ="y", 
        hline: bool =False, 
        yfmt=None) -> gg.ggplot:
    """
    Função que plota os gráficos usando ggplot (plotnine)
    """
    
    # Cria um gráfico de linha usando ggplot, formatando o datetime do eixo x
    fig = gg.ggplot(data=data) + gg.theme_light() +\
        gg.geom_line(mapping=gg.aes(y=y, x=x), size=0.5, color="#4d58bf") +\
        gg.scale_x_datetime(labels=lambda x: [dt.strftime("%d/%b/%Y") for dt in x])
    
    # Adiciona uma linha pontilhada em y=0.0 se hline for True
    if hline:
        fig = fig + gg.geom_hline(yintercept=0.0, linetype="dashed", size=1.3)
        
    # Formata o texto do eixo y se yfmt não for None
    if yfmt:
        fig = fig + gg.scale_y_continuous(
                labels=lambda y: [yfmt(yi) for yi in y]
        )
    
    # Adiciona os títulos e tema do gráfico (tira as linhas de grid, ajusta
    # o ângulo do eixo x, etc)
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


# Converte a data de início e fim da análise para string
start_str = start_dt.strftime("%b/%Y")
end_str = end_dt.strftime("%b/%Y")

# Plota os gráficos (são autoexplicativos)
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

df["log_selic"] = np.log(df["selic"])
chart(
      df, 
      "date", 
      "log_selic", 
      title=f"Taxa Selic de {start_str} a {end_str}",
      ylab="(Log) Taxa Selic (base 252 dias)",
      yfmt=lambda x: f"{np.exp(x)*100:.2f}%"
)


chart(
      df, 
      "date", 
      "rn", 
      title=f"Retorno Log-Preço nominal do Ibovespa de {start_str} a {end_str}",
      ylab="Retorno nominal (r + rf)",
      hline=True
)

chart(
      df, 
      "date", 
      "r", 
      title=f"Retorno Log-Preço Real do Ibovespa de {start_str} a {end_str}",
      ylab="Retorno real (r)",
      hline=True
)

chart(
      df, 
      "date", 
      "vn", 
      title=f"Volatilidade nominal de {start_str} a {end_str}",
      ylab="Volatilidade nominal |r + rf|",
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

chart(
      df, 
      "date", 
      "vi", 
      title=f"Volatilidade instantânea de {start_str} a {end_str}",
      ylab="Volatilidade instantânea",
      hline=True
)

chart(
      df, 
      "date", 
      "sd", 
      title=f"Volatilidade realizada (22 dias úteis) de {start_str} a {end_str}",
      ylab="Volatilidade realizada (22 dias úteis)",
      hline=True
)


# Cria uma função que calcula as autocorrelações
def autocorr(returns: pd.Series, n_lags:int = 100) -> List[float]:
    """
    Função que calcula as autocorrelações dos retornos
    """
    
    # Cria uma lista de correlações
    correlations = []
    for tau in range(n_lags):
        # Para cada tau, cria um dataframe contendo o valor dos retornos e o
        # valor dos retornos com um lag de -tau (t + tau)
        r_df = pd.DataFrame({
            "r": returns,
            "rlag": returns.shift(-tau)    
        })
        
        # Remove os valores que ficaram nulos
        r_df = r_df.dropna()
        
        # Calcula a covariância amostral 
        # cov(x,y) = soma((x-xm)*(y-ym)) / (n-1)
        cov = (r_df["r"] - r_df["r"].mean()) * (r_df["rlag"] - r_df["rlag"].mean())
        cov = cov.sum() / (len(cov) - 1.0)
        
        # Calcula a autocorrelação normalizada (coeficiente de correlação)
        # R(x,y) = cov(x,y) / (dp(x) * dp(y))
        corr = cov / (r_df["r"].std() * r_df["rlag"].std())
        
        # Adiciona o valor da correlação à lista de correlações
        correlations.append(float(corr))
    return correlations


# Cria uma função para plotar o gráfico da autocorrelação
def autocorr_chart(
        returns: pd.Series, 
        n_lags:int = 100,
        start:int = 0,
        title: str = "Title",
        ylab: str = "Autocorrelação",
        log=False) -> gg.ggplot:
    """
    Função que plota o gráfico das autocorrelações
    """
    
    # Se a escala é logarítmica o lag mínimo deve ser 1
    if log and start == 0:
        start = 1
    
    lags = list(range(n_lags))[start:]
    correlations = autocorr(returns, n_lags)[start:]
    
    # Se a escala é logarítmica, aplica log aos eixos x e y
    # não mostra a linha pontilhada em 0 se a escala é logarítmica
    if log:
        x = np.log(lags)
        y = np.log(correlations)
        hline=False
    else:
        x = lags
        y = correlations
        hline=True
    
    
    # Cria um gráfico de linhas usando ggplot 
    fig = gg.ggplot() + gg.theme_light() +\
        gg.geom_line(
            mapping=gg.aes(x=x, y=y), 
            size=0.5, 
            color="#4d58bf"
        )
    
    # Se a escala não é logarítmica, adiciona uma linha pontilhada passando por
    # y = 0
    if hline:
        fig = fig + gg.geom_hline(
            yintercept=0.0, 
            linetype="dashed", 
            size=0.5, 
            color="#383838"
        )  
    
    # Se a escala é logarítmica, exponenciaa os eixos x e y para mostrar os
    # valores originas do lag e da correlação
    if log:
        fig = fig + gg.scale_x_continuous(
            labels = lambda breaks: [f"{np.exp(br):.1f}" for br in breaks]
        ) +\
        gg.scale_y_continuous(
            labels = lambda breaks: [f"{np.exp(br):.2f}" for br in breaks]
        )
    
    # Define os títulos dos eixos e estilos do gráfico
    fig = fig + gg.ggtitle(title) +\
        gg.labs(
            x="Tempo (dias úteis)",
            y=ylab
        ) +\
        gg.theme(
           panel_border=gg.element_blank(),
           panel_grid=gg.element_blank(),
           axis_ticks_major_y=gg.element_blank(),
           plot_title=gg.element_text(hjust=0.5)
        )
    return fig


# Autocorrelações do retorno log-preço real
autocorr_chart(
    df["r"], 
    log=False, 
    n_lags=100,
    title=f"Autocorrelação normalizada do retorno do Ibovespa\n{start_str} a {end_str}",
    ylab="Autocorrelação do retorno real do Ibovespa (r)")

# Autocorrelações do retorno log-preço nominal
autocorr_chart(
    df["rn"], 
    log=False, 
    n_lags=100,
    title=f"Autocorrelação do retorno nominal de {start_str} a {end_str}",
    ylab="Autocorrelação do retorno nominal (r + rf)")

# Autocorrelaçoes da volatilidade real (módulo de r)
autocorr_chart(
    df["v"], 
    log=True, 
    n_lags=400,
    title=f"Autocorrelação normalizada da volatilidade do Ibovespa (log-log)\n{start_str} a {end_str}",
    ylab="Autocorrelação da volatilidade real |r|")

# Autocorrelaçoes da volatilidade nominal (módulo de rn)
autocorr_chart(
    df["vn"], 
    log=True, 
    n_lags=100,
    title=f"Autocorrelação da volatilidade nominal de {start_str} a {end_str} (log-log)",
    ylab="Autocorrelação da volatilidade nominal |r + rf|")


# -------------------------------------------------------------------
# Transformando a volatilidade usando Box-Cox
# -------------------------------------------------------------------

# Plota histograma da volatilidade original
sns.histplot(x=df["v"])

# Aplica Box-Cox à série df["v"], retorna valores transformados e λ estimado
df["v_bcx"], lambda_bcx = lamb=sp.stats.boxcox(df["v"])

# Plota histograma da volatilidade após Box-Cox
sns.histplot(x=df["v_bcx"])


def boxcox(x, lambd):
    """
    Função que aplica a transformação de Box-Cox manualmente
    """
    if lambd == 0.0:
        return np.log(x)
    return (np.power(x, lambd) - 1.0) / lambd


def inv_boxcox(x, lambd):
    """
    Função inversa da transformação de Box-Cox
    """
    if lambd == 0.0:
        return np.exp(x)
    return np.power(lambd*x + 1.0, 1.0/lambd)


# Calcula autocorrelação da volatilidade em até 400 defasagens
corrs = autocorr(df["v"], n_lags=400)

# Cria gráfico de autocorrelação com eixos na escala Box-Cox, mas rótulos 
# na escala original
gg.ggplot() + gg.theme_light() +\
    gg.geom_line(
        mapping=gg.aes(
            x=boxcox(np.arange(len(corrs)), lambda_bcx)[1:],
            y=boxcox(corrs, lambda_bcx)[1:]
        )    
    ) +\
    gg.ggtitle(f"Autocorrelação normalizada da volatilidade do Ibovespa\n{start_str} a {end_str}") +\
    gg.labs(
        y="Autocorrelação da volatilidade real |r|",
        x="Defasagem em dias úteis",
        subtitle=f"Plotagem Box-Cox—Box-Cox (λ={lambda_bcx:.3f})"
    ) +\
    gg.scale_x_continuous(
        labels = lambda breaks: [f"{inv_boxcox(br, lambda_bcx):.1f}" for br in breaks]
    ) +\
    gg.scale_y_continuous(
        labels = lambda breaks: [f"{inv_boxcox(br, lambda_bcx):.1f}" for br in breaks]
    ) +\
    gg.theme(
        plot_title=gg.element_text(hjust=0.5),
        plot_subtitle=gg.element_text(hjust=0.5),
        panel_grid=gg.element_blank()
    )
    

# Cria gráfico de autocorrelação com eixos na escala log-log, mas rótulos 
# na escala original
gg.ggplot() + gg.theme_light() +\
    gg.geom_line(
        mapping=gg.aes(
            x=np.log(np.arange(len(corrs)))[1:],
            y=np.log(corrs)[1:]
        )    
    ) +\
    gg.ggtitle(f"Autocorrelação normalizada da volatilidade do Ibovespa\n{start_str} a {end_str}") +\
    gg.labs(
        y="Autocorrelação da volatilidade real |r|",
        x="Defasagem em dias úteis",
        subtitle="Plotagem log-log"
    ) +\
    gg.scale_x_continuous(
        labels = lambda breaks: [f"{np.exp(br):.1f}" for br in breaks]
    ) +\
    gg.scale_y_continuous(
        labels = lambda breaks: [f"{np.exp(br):.1f}" for br in breaks]
    ) +\
    gg.theme(
        plot_title=gg.element_text(hjust=0.5),
        plot_subtitle=gg.element_text(hjust=0.5),
        panel_grid=gg.element_blank()
    )


# -------------------------------------------------------------------
# Transformando a volatilidade usando Box-Cox Imaginário
# -------------------------------------------------------------------

def boxcox_img(x, lambd, real=False):
    """
    Função que aplica a transformação de Box-Cox manualmente
    """
    if lambd == 0.0:
        return np.log(x)
    transf = np.array((np.power(x, lambd) - 1.0) / lambd)
    return transf.real if real else transf.imag


def inv_boxcox_img(x, lambd, real=False):
    """
    Função inversa da transformação de Box-Cox
    """
    if lambd == 0.0:
        return np.exp(x)
    transf = np.power(lambd*x + 1.0, 1.0/lambd)
    return transf.real if real else transf.imag


# Plota histograma da volatilidade original
sns.histplot(x=df["v"])

lambda_bcx = 0.24 + 40j
bx = boxcox_img(df["v"], lambda_bcx, real=True)
sns.histplot(x=bx)

# Calcula autocorrelação da volatilidade em até 400 defasagens
corrs = autocorr(df["v"], n_lags=400)

# Cria gráfico de autocorrelação com eixos na escala Box-Cox, mas rótulos 
# na escala original
gg.ggplot() + gg.theme_light() +\
    gg.geom_line(
        mapping=gg.aes(
            x=boxcox(np.arange(len(corrs)), lambda_bcx, real=False)[1:],
            y=boxcox(corrs, lambda_bcx, real=False)[1:]
        )    
    ) +\
    gg.ggtitle(f"Autocorrelação normalizada da volatilidade do Ibovespa\n{start_str} a {end_str}") +\
    gg.labs(
        y="Autocorrelação da volatilidade real |r|",
        x="Defasagem em dias úteis",
        subtitle=f"Plotagem Box-Cox—Box-Cox (λ={lambda_bcx:.3f})"
    ) +\
    gg.scale_x_continuous(
        labels = lambda breaks: [f"{inv_boxcox_img(br, lambda_bcx, real=False):.1f}" for br in breaks]
    ) +\
    gg.scale_y_continuous(
        labels = lambda breaks: [f"{inv_boxcox_img(br, lambda_bcx, real=False):.1f}" for br in breaks]
    ) +\
    gg.theme(
        plot_title=gg.element_text(hjust=0.5),
        plot_subtitle=gg.element_text(hjust=0.5),
        panel_grid=gg.element_blank()
    )

# Analisando a evolução de um investimento inicial de 1 R$ nos dois investimen-
# tos

# -------------------------------------------------------------------
# Função para reconstruir série de preços a partir de retornos logarítmicos
# -------------------------------------------------------------------
def prices_from_returns(returns, initial_price=1.0, return_initial=True):
    """
    Recebe uma sequência de retornos logarítmicos `returns` e um preço inicial,
    e retorna a série de preços correspondentes.
    
    Parâmetros:
    - returns: iterable de retornos log (float)
    - initial_price: preço inicial no tempo t=0 (default=1.0)
    - return_initial: se True, inclui o preço inicial na série; caso contrário,
      retorna apenas preços a partir do primeiro retorno.
    """
    prices = [initial_price] # lista que armazenará os preços, já iniciada com initial_price
    for r in returns:
        # exp(r) aplica a transformação inversa do log e multiplica pelo último preço
        price = np.exp(r) * prices[len(prices)-1]
        prices.append(price)
    # Retorna com ou sem o preço inicial, conforme flag
    return prices if return_initial else prices[1:]


# -------------------------------------------------------------------
# Construção do DataFrame de preços para BVSP e Selic
# -------------------------------------------------------------------
# Lista de datas extraída do DataFrame original
dates = list(df["date"])

# Inserção de um dia antes do primeiro registro para alinhar o preço inicial
day_before_second = df["date"][0] - timedelta(days=1.0)
dates.insert(0, day_before_second)
dates = np.array(dates)

# Gera o DataFrame com colunas de preços reconstrídos e datas
df_prices = pd.DataFrame({
    "Retorno BVSP": prices_from_returns(df["rn"]),
    "Retorno Selic": prices_from_returns(df["rf"]),
    "date": dates
})

# Converte para formato “long” para facilitar o uso no plotnine
df_prices_long = pd.melt(df_prices, id_vars="date")

# -------------------------------------------------------------------
# Gráfico da evolução de R$1,00 investido em BVSP vs. Selic
# -------------------------------------------------------------------
fig = gg.ggplot(data = df_prices_long) + gg.theme_light() +\
    gg.geom_line(
        mapping = gg.aes(
            x = "date",
            y = "value",
            color = "variable"
        )
    ) +\
    gg.ggtitle(
        title=f"Evolução de R$ 1,00 investido de {start_str} a {end_str}"
    ) +\
    gg.labs(
        y="Valor (R$)"
    ) +\
    gg.scale_y_continuous(
        labels=lambda x: [f"R$ {xi:.2f}".replace(".", ",") for xi in x]    
    ) +\
    gg.scale_x_datetime(
        labels=lambda x: [dt.strftime("%b/%Y") for dt in x]
    ) +\
    gg.theme(
       panel_border=gg.element_blank(),
       panel_grid_minor=gg.element_blank(),
       panel_grid_major_x=gg.element_blank(),
       axis_ticks_major_y=gg.element_blank(),
       axis_text_x=gg.element_text(angle=30),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       legend_key=gg.element_blank(),
       plot_title=gg.element_text(hjust=0.5)
    ) +\
    gg.scale_color_manual(
        values={
            "Retorno BVSP": "#bd4b22",
            "Retorno Selic": "#09d616"
        }
    )
fig


# Comparando pedaços da série temporal dos retornos

# -------------------------------------------------------------------
# Histograma dos retornos log-preço do BVSP (Comparação com a Normal)
# -------------------------------------------------------------------
r_max = df["r"].std() * 4

x_dist = np.linspace(-r_max, r_max, 10000)
pdf_dist = sp.stats.norm.pdf(x_dist, loc=df["r"].mean(), scale=df["r"].std())

gg.ggplot() + gg.theme_light() +\
    gg.geom_histogram(
        mapping=gg.aes(
            x=df["r"],
            y=gg.after_stat("density")
        ),
        fill="#a8a8a8"
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x=x_dist,
            y=pdf_dist
        )
    ) +\
    gg.scale_y_log10() +\
    gg.ggtitle(
        title=f"Histograma do Retorno Log-Preço do BVSP de {start_str} a {end_str}"
    ) +\
    gg.labs(
        y="Densidade de Probabilidade",
        x="Retorno Log-Preço do BVSP descontado (r)",
        subtitle="Comparação com a Normal de mesma média e variância"
    ) +\
    gg.coord_cartesian(xlim=[-r_max, r_max]) +\
    gg.theme(
       panel_grid=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5),
       plot_subtitle=gg.element_text(hjust=0.5)        
    )
# -------------------------------------------------------------------
# Histograma dos retornos log-preço do BVSP (Comparação com a Cauchy)
# -------------------------------------------------------------------
r_max = df["r"].std() * 4

x_dist = np.linspace(-r_max, r_max, 10000)
pdf_dist = sp.stats.cauchy.pdf(x_dist, loc=df["r"].mean(), scale=0.011)

gg.ggplot() + gg.theme_light() +\
    gg.geom_histogram(
        mapping=gg.aes(
            x=df["r"],
            y=gg.after_stat("density")
        ),
        fill="#a8a8a8"
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x=x_dist,
            y=pdf_dist
        )
    ) +\
    gg.ggtitle(
        title=f"Histograma do Retorno Log-Preço do BVSP de {start_str} a {end_str}"
    ) +\
    gg.scale_y_log10() +\
    gg.labs(
        y="Densidade de Probabilidade",
        x="Retorno Log-Preço do BVSP descontado (r)",
        subtitle="Comparação com a Cauchy com valor de γ=0.011"
    ) +\
    gg.coord_cartesian(xlim=[-r_max, r_max]) +\
    gg.theme(
       panel_grid=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5),
       plot_subtitle=gg.element_text(hjust=0.5)        
    )
    
# -------------------------------------------------------------------
# Histograma da volatilidade do BVSP (Comparação com a Levy)
# -------------------------------------------------------------------
r_max = df["v"].std() * 4

x_dist = np.linspace(0.0, r_max, 10000)
pdf_dist = sp.stats.levy.pdf(x_dist, scale=0.008)

gg.ggplot() + gg.theme_light() +\
    gg.geom_histogram(
        mapping=gg.aes(
            x=df["v"],
            y=gg.after_stat("density")
        ),
        fill="#a8a8a8"
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x=x_dist,
            y=pdf_dist
        )
    ) +\
    gg.ggtitle(
        title=f"Histograma da volatilidade do BVSP de {start_str} a {end_str}"
    ) +\
    gg.labs(
        y="Densidade de Probabilidade",
        x="Volatilidade do BVSP",
        subtitle="Comparação com a Lévy com valor de c=0.008"
    ) +\
    gg.coord_cartesian(xlim=[0.0 - df["v"].std()*0.5, r_max]) +\
    gg.theme(
       panel_grid=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5),
       plot_subtitle=gg.element_text(hjust=0.5)        
    )
    

# -------------------------------------------------------------------
# Histograma da volatilidade do BVSP (Comparação com a Exponencial)
# -------------------------------------------------------------------
r_max = df["v"].std() * 4

x_dist = np.linspace(0.0, r_max, 10000)
pdf_dist = sp.stats.expon.pdf(x_dist, scale=0.016)

gg.ggplot() + gg.theme_light() +\
    gg.geom_histogram(
        mapping=gg.aes(
            x=df["v"],
            y=gg.after_stat("density")
        ),
        fill="#a8a8a8"
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x=x_dist,
            y=pdf_dist
        )
    ) +\
    gg.ggtitle(
        title=f"Histograma da volatilidade do BVSP de {start_str} a {end_str}"
    ) +\
    gg.labs(
        y="Densidade de Probabilidade",
        x="Volatilidade do BVSP",
        subtitle="Comparação com a Exponencial com valor de γ=0.016"
    ) +\
    gg.coord_cartesian(xlim=[0.0 - df["v"].std()*0.5, r_max]) +\
    gg.theme(
       panel_grid=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5),
       plot_subtitle=gg.element_text(hjust=0.5)        
    )
    

# -------------------------------------------------------------------
# Histograma da volatilidade do BVSP (Comparação com a Log-Normal)
# -------------------------------------------------------------------
r_max = df["v"].std() * 4

x_dist = np.linspace(0.0, r_max, 10000)
pdf_dist = sp.stats.lognorm.pdf(x_dist, s=1.0, scale=df["v"].std())

gg.ggplot() + gg.theme_light() +\
    gg.geom_histogram(
        mapping=gg.aes(
            x=df["v"],
            y=gg.after_stat("density")
        ),
        fill="#a8a8a8"
    ) +\
    gg.geom_line(
        mapping = gg.aes(
            x=x_dist,
            y=pdf_dist
        )
    ) +\
    gg.ggtitle(
        title=f"Histograma da volatilidade do BVSP de {start_str} a {end_str}"
    ) +\
    gg.labs(
        y="Densidade de Probabilidade",
        x="Volatilidade do BVSP",
        subtitle="Comparação com a Log-Normal com mesma média e desvio padrão"
    ) +\
    gg.coord_cartesian(xlim=[0.0 - df["v"].std()*0.5, r_max]) +\
    gg.theme(
       panel_grid=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5),
       plot_subtitle=gg.element_text(hjust=0.5)        
    )
    

# -------------------------------------------------------------------
# Seleção de dois recortes da série de retornos e comparação
# -------------------------------------------------------------------
# Recortes em índices fixos
df_slice1 = df.loc[:50,:].copy()
df_slice2 = df.loc[150:200,:].copy()

# Converte para listas simples de floats
slice1 = [float(x) for x in df_slice1["r"]]
slice2 = [float(x) for x in df_slice2["r"]]

# DataFrame “long” para sobrepor as duas séries no mesmo gráfico
df_slices = pd.DataFrame({
    "index": [x for x in range(len(slice1))] + [x for x in range(len(slice2))],
    "variable": (["Recorte 1"] * len(slice1)) + (["Recorte 2"] * len(slice2)),
    "value": slice1 + slice2    
})

# Plota os recortes dos retornos
chart(data=df_slice1, x="date", y="r")
chart(data=df_slice2, x="date", y="r")

# Teste estatístico de Kolmogorov-Smirnov entre os dois recortes
_, p = kstest(slice1, slice2)
print(p)

# Gráfico comparativo dos recortes
fig = gg.ggplot(data=df_slices) + gg.theme_light() +\
    gg.geom_line(
        mapping=gg.aes(
            x="index",
            y="value",
            color="variable"
        )
    ) +\
    gg.scale_color_manual(
        values={
            "Recorte 1": "#c95e28",
            "Recorte 2": "blue"
        }    
    ) +\
    gg.ggtitle(
        title="Comparação entre os dois recortes do Retorno Log-Preço"
    ) +\
    gg.labs(
        y="Retorno Log-Preço do BVSP descontado (r)"
    ) +\
    gg.theme(
       panel_border=gg.element_blank(),
       panel_grid=gg.element_blank(),
       axis_ticks=gg.element_blank(),
       axis_text_x=gg.element_blank(),
       axis_title_x=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="bottom",
       plot_title=gg.element_text(hjust=0.5)
    )
fig
    
# -------------------------------------------------------------------
# Construção de splits e análise de p-valores entre todos os pares
# -------------------------------------------------------------------
def get_splits(x, length):
    """
    Divide o vetor x em pedaços de tamanho `length`.
    A última parte pode ter tamanho menor se len(x) não for múltiplo de length.
    """
    return [x[i : i + length] for i in range(0, len(x), length)]

# Cria os splits de tamanho 50
splits = get_splits(df["r"], 50)

# Calcula p-valores de K-S para cada par de splits (i ≠ j)
pvalues = list()
for i in range(len(splits)):
    for j in range(len(splits)):
        if i==j: continue
        _, p = kstest(splits[i], splits[j])
        pvalues.append(p)
pvalues = np.array(pvalues)

# Contagem de quantos testes rejeitam ou não H0 em diferentes níveis de significância
df_pvalues = pd.DataFrame({
    "classe": ["Não Rejeita H0", "Rejeita a 10%", "Rejeita a 5%", "Rejeita a 1%"],
    "quantidade": [
        len(pvalues[pvalues > 0.1]),
        len(pvalues[pvalues <= 0.1]),
        len(pvalues[pvalues <= 0.05]),
        len(pvalues[pvalues <= 0.01]), 
    ]
})

# Gráfico de colunas dos resultados do K-S test
fig = gg.ggplot(df_pvalues) + gg.theme_light() +\
    gg.geom_col(
        mapping = gg.aes(
            x="classe",
            y="quantidade",
            fill="classe"
        ) 
    ) +\
    gg.scale_x_discrete(
        limits=[
            "Não Rejeita H0",
            "Rejeita a 10%",
            "Rejeita a 5%",
            "Rejeita a 1%"
        ]
    ) +\
    gg.ggtitle(
        title="H0: Os dois recortes vêm da mesma população"
    ) +\
    gg.labs(
        y="Número de observações"
    ) +\
    gg.theme(
       panel_border=gg.element_blank(),
       panel_grid=gg.element_blank(),
       axis_ticks=gg.element_blank(),
       axis_title_x=gg.element_blank(),
       legend_title=gg.element_blank(),
       legend_position="none",
       plot_title=gg.element_text(hjust=0.5)
    )
fig


# -------------------------------------------------------------------
# Recortando um pedaço da série e colocando na mesma escala da série
# original
# -------------------------------------------------------------------

st_slice = datetime(year=2018, month=1, day=1)
end_slice = datetime(year=2025, month=4, day=27)

str_st_slice = st_slice.strftime("%d/%b/%Y")
str_end_slice = end_slice.strftime("%d/%b/%Y")

df_slice1 = df[(df["date"] > st_slice) & (df["date"] < end_slice)].copy()

df["axis"] = np.arange(len(df), dtype="float32")
df_slice1["axis"] = np.arange(len(df_slice1), dtype="float32")
df_slice1["axis"] = df_slice1["axis"] * (np.max(df["axis"]) / np.max(df_slice1["axis"]))
df_slice1["r"] = df_slice1["r"] * (np.std(df["r"]) / np.std(df_slice1["r"]))
df_slice1["r"] = df_slice1["r"] + (df["r"].mean() - df_slice1["r"].mean())

y_min, y_max = df["r"].min(), df["r"].max()
y_max = np.max([np.abs(y_min), np.abs(y_max)])
y_max = np.ceil(y_max*100.0)/100
ticks = np.linspace(-y_max, y_max, 5)

legends = (
    ["Série Total" for x in range(len(df))],
    ['Recorte' for x in range(len(df_slice1))]    
)
gg.ggplot() + gg.theme_light() +\
    gg.geom_line(
        mapping=gg.aes(
            x=df["axis"],
            y=df["r"],
            color=legends[0]
        ),
        size=0.5
    ) +\
    gg.geom_line(
        mapping=gg.aes(
            x=df_slice1["axis"],
            y=df_slice1["r"],
            color=legends[1]
        ),
        size=0.5
    ) +\
    gg.ggtitle(
        "Série total versus recorte: comparação de retornos na mesma escala"
    ) +\
    gg.labs(
        y = "Retorno Log-Preço",
        subtitle=f"Recorte: {str_st_slice} a {str_end_slice}"
    ) +\
    gg.scale_y_continuous(
        breaks=ticks,
        limits=[-y_max, y_max]
    ) +\
    gg.scale_color_manual(
        name="ignored",
        values={
            "Série Total": "#c9c9c9",
            "Recorte": "#4d3cc9"
        }
    ) +\
    gg.theme(
        panel_border=gg.element_blank(),
        panel_grid=gg.element_blank(),
        axis_text_x=gg.element_blank(),
        axis_title_x=gg.element_blank(),
        axis_ticks=gg.element_blank(),
        plot_title=gg.element_text(hjust=0.5),
        plot_subtitle=gg.element_text(hjust=0.5),
        legend_position="bottom",
        legend_title = gg.element_blank(),
        legend_key = gg.element_blank()
    )
