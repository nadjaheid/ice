# # Projeto Integrado I: O caso Loja Ice

# # Previsões para as vendas de videogame da Loja Online Ice para 2017

# ## Importação do arquivo de dados e informações gerais


import pandas as pd # importando a bibioteca pandas
import matplotlib.pyplot as plt # importando a biblioteca matplotlib
import seaborn as sns # importando a biblioteca seaborn
import scipy.stats as stats # importando a biblioteca SciPy
import numpy as np # importando a biblioteca Numpy


games = pd.read_csv('/datasets/games.csv') # abrindo o arquivo de dados


games.info() # acessando as principais informações do dataframe


print(games.sample(10)) #extraindo uma amostra aleatória de 10 linhas do dataframe


# Precisam ser transformadas as variáveias 'Year_of_Release', 'User_Score' para int64 e float64, respectivamente. Uma vez que o ano é um dado de valor inteiro e a pontuação do usuário um dado numérico de ponto flutuante. Existem dados faltantes que precisam ser trabalhados. Os nomes das colunas precisam ser passados para letra minúscula, para se evitar erros de nomenclatura ao se chamar os dados.

# ## Preparação dos dados

# Transformando os nomes das colunas em letra minúscula
games.columns = games.columns.str.lower()
nomes_colunas = games.columns
print("Nomes das Colunas:")
print(nomes_colunas)



# Encontrando os valores nulos
print(games.isnull().sum()/len(games))


# como a coluna 'year_of_release' será importante para a analise das vendas da empresa e são poucos os dados sem o ano de lançamento, optei por remover estas linhas.

games=games.dropna(subset=['year_of_release'])


# Os valores nulos para as variáveis do tipo string, substituí por None. 
# 
# Para os dados de pontuação ('critic_score' e 'user_score'), substituí pelo valor 999, uma vez que são jogos que aparentemente não tiveram vendas expressivas, e por isso mesmo não foram avaliados, mas que podem ser importantes de serem analisados para se compreender o motivo das vendas baixas.

games['name'].fillna('None', inplace=True)
games['genre'].fillna('None', inplace=True)
games['user_score'].fillna('999', inplace=True) 
games['critic_score'].fillna(999, inplace=True)
games['rating'].fillna('None', inplace=True)
print(games.isnull().sum())


# Convertendo a coluna 'year_of_release' para int64
games['year_of_release'] = games['year_of_release'].astype('int64') 


# Verificando quais os valores únicos na coluna 'user_score'

print(games['user_score'].unique())


# Como existe um valor não numerico 'tbd'(que provavelmente se refere a críticas ainda não inseridas), vou transformá-lo em 999, antes de proceder à conversão dos dados para numéricos.


games['user_score'] = games['user_score'].replace('tbd', '999')
print(games['user_score'].unique())


# Convertendo a coluna 'user_score' para float64
games['user_score'] = games['user_score'].astype('float64')


# Como passo final na preparação dos dados, criei uma nova coluna chamada 'total_sales', que é o total de vendas para cada jogo.
games['total_sales'] = games[['na_sales', 'eu_sales', 'jp_sales', 'other_sales']].sum(axis=1)
games.info()


# ### Conclusão
# Nesta fase de informação e preparação dos dados perdebeu-se que alguns jogos não tinham informação sobre o ano de lançamento, foram excluídas do banco de dados. Boa parte dos valores nulos restantes se encontravam em dados relacionados às pontuações dadas pelos usuários e pela crítica, justamente naquelas observações onde as vendas haviam sido mínimas ou nulas. Nestes casos, optei por preencher com 999 para identificar quais eram os jogos que tinhas baixas vendas, para caso fosse necessário para análise futura. 

# ## Análise dos dados

# ### Lançamentos por ano

# Calculando a contagem de jogos por ano
contagem_por_ano = games['year_of_release'].value_counts()

# Plotando o gráfico de barras
plt.figure(figsize=(10, 6))
contagem_por_ano.sort_index().plot(kind='bar', color='skyblue')
plt.title('Número de Jogos Lançados por Ano')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Número de Jogos')
plt.show()


# Pelo gráfico exposto, percebe-se que a partir dos anos 2000 houve um aumento expressivo no lançamento de jogos no mercado. Até o ano de 1994 menos de duzentos jogos eram lançados, entre 1994 e 2000 eram lançados cerca de 400 jogos por ano. Até 2005, esse número saltou para 800, até chegar no máximo em 2008 e 2009 com cerca de 1400 jogos lançados por ano. Após esse período reduziu, chegando a 600 jogos por ano, entre 2012 e 2016.

# ### Variação de vendas entre as plataformas

# Nesta seção será visto como as vendas variaram de plataforma para plataforma. Para as plataformas com as maiores vendas totais será construída uma distribuição com base em dados para cada ano. 
# 
# Depois, serão exibidas as plataformas que costumavam ser populares, mas agora não têm vendas.


# Agrupando por plataforma e calculando as vendas totais
vendas_por_plataforma = games.groupby('platform')['total_sales'].sum()

# Ordenando as plataformas pelas vendas totais em ordem decrescente
plataformas_populares = vendas_por_plataforma.sort_values(ascending=False)

# Selecionando as top 5 plataformas com maiores vendas totais
top_plataformas = plataformas_populares.head(5)
print(top_plataformas)


# Construindo uma distribuição com base nos dados para cada ano (considerando apenas as top 5 plataformas)
games_top_plataformas = games[games['platform'].isin(top_plataformas.index)]

# Agrupando por ano e calculando as vendas totais
vendas_por_ano_top_plataformas = games_top_plataformas.groupby('year_of_release')['total_sales'].sum()

# Plotando um gráfico de barras para a distribuição das vendas por ano
vendas_por_ano_top_plataformas.plot(kind='bar', figsize=(10, 6), title='Vendas Totais por Ano (Top 5 Plataformas)')
plt.xlabel('Ano de Lançamento')
plt.ylabel('Vendas Totais (milhões)')
plt.show()

# Limite para vendas totais consideradas "muito baixas"
limite_vendas_baixas = 1.0  # Ajuste conforme necessário

# Filtrando o DataFrame para incluir apenas dados de plataformas com lançamentos antes de 1995
plataformas_antes_1995 = games[games['year_of_release'] < 1995]

# Calculando as vendas totais por plataforma para essas plataformas
vendas_por_plataforma_antes_1995 = plataformas_antes_1995.groupby('platform')['total_sales'].sum()

# Identificando plataformas com vendas totais inferiores ao limite considerado "muito baixas"
plataformas_baixas_vendas_antes_1995 = vendas_por_plataforma_antes_1995[vendas_por_plataforma_antes_1995 < limite_vendas_baixas].index

print(f"Plataformas com lançamentos antes de 1995 e vendas muito baixas: {plataformas_baixas_vendas_antes_1995}")


# Identificando os anos de lançamento da primeira e última ocorrência de cada plataforma
anos_primeira_aparicao = games.groupby('platform')['year_of_release'].min()
anos_ultima_aparicao = games.groupby('platform')['year_of_release'].max()

# Calculando a diferença de anos entre a última e primeira aparição de cada plataforma
tempo_vida_plataformas = anos_ultima_aparicao - anos_primeira_aparicao

# Exibindo a média e a mediana do tempo de vida das plataformas
print(f"Média do tempo de vida das plataformas: {tempo_vida_plataformas.mean():.2f} anos")
print(f"Mediana do tempo de vida das plataformas: {tempo_vida_plataformas.median()} anos")


# Diante do tempo médio de vida de 7 anos e de mediana de 6 anos para cada plataforma e como houve uma queda expressiva no lançamento de novos jogos a partir de 2011, serão considerados apenas os dados do período de 2012 a 2016, para as previsões para 2017.

# Filtrando o DataFrame para incluir apenas dados do período de 2012 a 2016
games_novo = games.query('2012 <= year_of_release <= 2016')

# Calculando as vendas totais por plataforma em 2016 no DataFrame 'games_novo'
vendas_por_plataforma_2016 = games_novo.groupby('platform')['total_sales'].sum()

# Ordenando as plataformas pelas vendas totais em 2016 em ordem decrescente
plataformas_lideres_2016 = vendas_por_plataforma_2016.sort_values(ascending=False)

print(f"Plataformas líderes em vendas em 2016: {plataformas_lideres_2016}")

# Calcular as vendas médias por plataforma no DataFrame 'games_novo'
vendas_medias_por_plataforma = games_novo.groupby('platform')['total_sales'].mean()

# Calcular a variação percentual nas vendas médias por plataforma em relação ao ano anterior
variacao_percentual_media = vendas_medias_por_plataforma.pct_change()

# Identificar plataformas que estão crescendo ou diminuindo em média (considerando um limiar de 10% de variação)
plataformas_crescendo_media = variacao_percentual_media[variacao_percentual_media > 0.1].index
plataformas_diminuindo_media = variacao_percentual_media[variacao_percentual_media < -0.1].index

print(f"Plataformas crescendo em média: {plataformas_crescendo_media}")
print(f"Plataformas diminuindo em média: {plataformas_diminuindo_media}")

# Filtrando plataformas que têm vendas totais em 2016 e que estão crescendo
plataformas_potencialmente_lucrativas = set(plataformas_lideres_2016.index) & set(plataformas_crescendo_media)

print(f"Plataformas potencialmente lucrativas: {plataformas_potencialmente_lucrativas}")

plt.figure(figsize=(15, 8))
sns.boxplot(x='platform', y='total_sales', data=games_novo, showfliers=False)
plt.title('Diagrama de Caixa das Vendas Globais por Plataforma')
plt.xlabel('Plataforma')
plt.ylabel('Vendas Globais (milhões)')
plt.show()


# Pode-se perceber que as três plataformas líderes de vendas em 2016 (PS4, PS3 e X360) estão entre as que potencialmente podem alferir lucros. Aparece também uma de menor expressividade que é a PSV.
# Chama a atenção a X360, por ter a maior variabilidade nas vendas, mas também apresentar a maior mediana.
# As plataformas PS4 e XOne apresentam vendas parecidas, entretanto, a PS4 se mostrou promissora enquanto que a XOne aparentou ter decrescimento. 

# ### Relação entre avaliações e vendas

# Filtrar o DataFrame para incluir apenas os dados da plataforma PS4
ps4_data = games_novo[games_novo['platform'] == 'PS4']


# Filtrar as linhas onde 'user_score' e 'critic_score' não são 999
filtered_games = ps4_data[(ps4_data['user_score'] != 999) & (ps4_data['critic_score'] != 999)]

# Criar gráfico de dispersão para avaliação de usuário vs. vendas totais
plt.figure(figsize=(10, 6))
plt.scatter(filtered_games['user_score'], filtered_games['total_sales'], color='blue', alpha=0.5)
plt.title('Avaliação de Usuário vs. Vendas Totais')
plt.xlabel('Avaliação de Usuário')
plt.ylabel('Vendas Totais (milhões)')
plt.show()

# Calcular correlação entre avaliação de usuário e vendas totais
correlation_user_sales = np.corrcoef(filtered_games['user_score'], filtered_games['total_sales'])[0, 1]
print(f"Correlação entre avaliação de usuário e vendas totais: {correlation_user_sales}")

# Criar gráfico de dispersão para avaliação de crítico vs. vendas totais
plt.figure(figsize=(10, 6))
plt.scatter(filtered_games['critic_score'], filtered_games['total_sales'], color='red', alpha=0.5)
plt.title('Avaliação de Crítico vs. Vendas Totais')
plt.xlabel('Avaliação de Crítico')
plt.ylabel('Vendas Totais (milhões)')
plt.show()

# Calcular correlação entre avaliação de crítico e vendas totais
correlation_critic_sales = np.corrcoef(filtered_games['critic_score'], filtered_games['total_sales'])[0, 1]
print(f"Correlação entre avaliação de crítico e vendas totais: {correlation_critic_sales}")


# Aparentemente, os usurário do PS4 não analisam as críticas dos usuários para suas decisões de compra, mas analisam as críticas profissionais (apesar da correlação entre estas duas variáveis ter sido relativamente baixa, se mostrou positiva.
# 
# Analisando abaixo para a plataforma Xbox One pode-se ver que esse padrão se repete.


# Filtrar as linhas onde 'user_score' e 'critic_score' não são 999 para a plataforma Xbox One
xbox_one_games = games[(games['platform'] == 'XOne') & (games['user_score'] != 999) & (games['critic_score'] != 999)]

# Adicionar gráfico de dispersão para avaliação de usuário vs. vendas totais na plataforma Xbox One
plt.scatter(xbox_one_games['user_score'], xbox_one_games['total_sales'], color='red', alpha=0.5, label='Xbox One')

# Calcular correlação entre avaliação de usuário e vendas totais na plataforma Xbox One
correlation_user_sales_xbox_one = np.corrcoef(xbox_one_games['user_score'], xbox_one_games['total_sales'])[0, 1]
print(f"Correlação entre avaliação de usuário e vendas totais (Xbox One): {correlation_user_sales_xbox_one}")

# Adicionar legenda
plt.legend()
plt.show()


# Adicionar gráfico de dispersão para avaliação de profissional vs. vendas totais na plataforma Xbox One
plt.scatter(xbox_one_games['critic_score'], xbox_one_games['total_sales'], color='red', alpha=0.5, label='Xbox One')

# Calcular correlação entre avaliação de usuário e vendas totais na plataforma Xbox One
correlation_user_sales_xbox_one = np.corrcoef(xbox_one_games['critic_score'], xbox_one_games['total_sales'])[0, 1]
print(f"Correlação entre avaliação de profissionais e vendas totais (Xbox One): {correlation_user_sales_xbox_one}")

# Adicionar legenda
plt.legend()
plt.show()


# ### Relação entre gênero e vendas


# Calcular as vendas totais por gênero
vendas_por_genero = games_novo.groupby('genre')['total_sales'].sum().sort_values(ascending=False)

# Calcular a participação percentual de cada gênero nas vendas totais
participacao_percentual = (vendas_por_genero / vendas_por_genero.sum()) * 100

# Plotar um gráfico de barras para as vendas totais por gênero
plt.figure(figsize=(12, 6))
sns.barplot(x=vendas_por_genero.index, y=vendas_por_genero.values, palette="viridis")
plt.title('Vendas Totais por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Vendas Globais (milhões)')
plt.xticks(rotation=45, ha='right')  # Rotacionar os rótulos para melhor legibilidade

# Adicionar um segundo eixo y para a participação percentual
ax2 = plt.gca().twinx()
ax2.set_ylabel('Participação Percentual (%)', color='red')
ax2.plot(vendas_por_genero.index, participacao_percentual, color='red', marker='o', linestyle='dashed')

# Exibir o gráfico
plt.show()


#  O gênero que os usuários mais gostam é Action (50% das vendas). Se somarmos os 4 principais gêneros Action, Shooter, Role-Playing e Sports chegamos a cerca de 80% das vendas.

# ### Conclusões

# Os lançamentos de jogos alcançaram o auge entre 2008 e 2009, apresentando uma queda acentuada após isso, e permanecendo no patamar de 600 jogos por ano, a partir de 2012.
# As 3 principais plataformas que se destacaram nos últimos anos e se mostram promissoras são PS4, PS3 e X360.
# Os usuários de jogos parecem levar em conta as críticas profissinais para a aquisição de jogos, e preferem os gêneros Action, Shooter, Role-Playing e Sports, respectivamente.

# ## Criação de um perfil de usuário para cada região

# ### América do Norte

# Filtrar os dados para a Região da América do Norte (NA)
na_data = games_novo[['platform', 'na_sales']].groupby('platform').sum().sort_values(by='na_sales', ascending=False).head(5)

# Plotar um gráfico de barras para as vendas totais por plataforma na América do Norte
plt.figure(figsize=(10, 6))
sns.barplot(x=na_data.index, y='na_sales', data=na_data, palette="Blues")
plt.title('Vendas Totais por Plataforma na América do Norte')
plt.xlabel('Plataforma')
plt.ylabel('Vendas Totais (milhões)')
plt.show()

# Filtrar os dados para incluir apenas a região da América do Norte
na_data = games_novo[games_novo['na_sales'] > 0]

# Calcular as vendas totais por gênero na América do Norte
vendas_por_genero_na = na_data.groupby('genre')['na_sales'].sum().sort_values(ascending=False)

# Exibir os cinco principais gêneros na América do Norte
top_generos_na = vendas_por_genero_na.head(5)
print("Cinco principais gêneros na América:")
print(top_generos_na)

# Plotar um gráfico de barras para visualizar a distribuição das vendas por gênero na América do Norte
plt.figure(figsize=(10, 6))
sns.barplot(x=top_generos_na.index, y=top_generos_na.values, palette="viridis")
plt.title('Vendas Totais por Gênero na América do Norte')
plt.xlabel('Gênero')
plt.ylabel('Vendas Globais (milhões)')
plt.xticks(rotation=45, ha='right')
plt.show()


# Calcular as vendas totais por classificação do ESRB na América do Norte
vendas_por_classificacao_na = games_novo.groupby('rating')['na_sales'].sum().sort_values(ascending=False)

# Exibir as vendas por classificação na América do Norte
print("Vendas por classificação na América do Norte:")
print(vendas_por_classificacao_na)

# Plotar um gráfico de barras para visualizar as vendas por classificação na América do Norte
plt.figure(figsize=(8, 6))
sns.barplot(x=vendas_por_classificacao_na.index, y=vendas_por_classificacao_na.values, palette="viridis")
plt.title('Vendas Totais por Classificação do ESRB na América do Norte')
plt.xlabel('Classificação do ESRB')
plt.ylabel('Vendas Globais (milhões)')
plt.show()


# ### Europa

# Filtrar os dados para a Região da União Europeia (UE)
eu_data = games_novo[['platform', 'eu_sales']].groupby('platform').sum().sort_values(by='eu_sales', ascending=False).head(5)

# Plotar um gráfico de barras para as vendas totais por plataforma na União Europeia
plt.figure(figsize=(10, 6))
sns.barplot(x=eu_data.index, y='eu_sales', data=eu_data, palette="Reds")
plt.title('Vendas Totais por Plataforma na União Europeia')
plt.xlabel('Plataforma')
plt.ylabel('Vendas Totais (milhões)')
plt.show()


# Filtrar os dados para incluir apenas a região da Europa
eu_data = games_novo[games_novo['eu_sales'] > 0]

# Calcular as vendas totais por gênero na Europa
vendas_por_genero_eu = eu_data.groupby('genre')['eu_sales'].sum().sort_values(ascending=False)

# Exibir os cinco principais gêneros na Europa
top_generos_eu = vendas_por_genero_eu.head(5)
print("Cinco principais gêneros na Europa:")
print(top_generos_eu)

# Plotar um gráfico de barras para visualizar a distribuição das vendas por gênero na Europa
plt.figure(figsize=(10, 6))
sns.barplot(x=top_generos_eu.index, y=top_generos_eu.values, palette="viridis")
plt.title('Vendas Totais por Gênero na Europa')
plt.xlabel('Gênero')
plt.ylabel('Vendas Globais (milhões)')
plt.xticks(rotation=45, ha='right')
plt.show()


# Calcular as vendas totais por classificação do ESRB na Europa
vendas_por_classificacao_eu = games_novo.groupby('rating')['eu_sales'].sum().sort_values(ascending=False)

# Exibir as vendas por classificação na Europa
print("Vendas por classificação na Europa:")
print(vendas_por_classificacao_eu)

# Plotar um gráfico de barras para visualizar as vendas por classificação na Europa
plt.figure(figsize=(8, 6))
sns.barplot(x=vendas_por_classificacao_na.index, y=vendas_por_classificacao_eu.values, palette="viridis")
plt.title('Vendas Totais por Classificação do ESRB na Europa')
plt.xlabel('Classificação do ESRB')
plt.ylabel('Vendas Globais (milhões)')
plt.show()


# ### Japão

# Filtrar os dados para a Região Japonesa (JP)
jp_data = games_novo[['platform', 'jp_sales']].groupby('platform').sum().sort_values(by='jp_sales', ascending=False).head(5)

# Plotar um gráfico de barras para as vendas totais por plataforma no Japão
plt.figure(figsize=(10, 6))
sns.barplot(x=jp_data.index, y='jp_sales', data=jp_data, palette="Greens")
plt.title('Vendas Totais por Plataforma no Japão')
plt.xlabel('Plataforma')
plt.ylabel('Vendas Totais (milhões)')
plt.show()

# Filtrar os dados para incluir apenas a região do Japão
jp_data = games_novo[games_novo['jp_sales'] > 0]

# Calcular as vendas totais por gênero do Japão
vendas_por_genero_jp = jp_data.groupby('genre')['jp_sales'].sum().sort_values(ascending=False)

# Exibir os cinco principais gêneros na Europa
top_generos_jp = vendas_por_genero_jp.head(5)
print("Cinco principais gêneros no Japão:")
print(top_generos_jp)

# Plotar um gráfico de barras para visualizar a distribuição das vendas por gênero na Europa
plt.figure(figsize=(10, 6))
sns.barplot(x=top_generos_jp.index, y=top_generos_jp.values, palette="viridis")
plt.title('Vendas Totais por Gênero no Japão')
plt.xlabel('Gênero')
plt.ylabel('Vendas Globais (milhões)')
plt.xticks(rotation=45, ha='right')
plt.show()

# Calcular as vendas totais por classificação do ESRB no Japão
vendas_por_classificacao_jp = games_novo.groupby('rating')['eu_sales'].sum().sort_values(ascending=False)

# Exibir as vendas por classificação no Japão
print("Vendas por classificação na Japão:")
print(vendas_por_classificacao_jp)

# Plotar um gráfico de barras para visualizar as vendas por classificação no Japão
plt.figure(figsize=(8, 6))
sns.barplot(x=vendas_por_classificacao_jp.index, y=vendas_por_classificacao_jp.values, palette="viridis")
plt.title('Vendas Totais por Classificação do ESRB no Japão')
plt.xlabel('Classificação do ESRB')
plt.ylabel('Vendas Globais (milhões)')
plt.show()


# ### Conclusões

# As preferências dos usuários dos principais mercados, América do Norte, Europa e Japão, diferem em alguns aspectos:
# - Na América do norte as principais plataformas utilizadas foram X360, PS4, PS3 e XOne, sendo que as três últimas dividem o mercado de uma maneira mais equitativa. Os gêneros preferidos são Action e Shooter.
# - Na Europa, as principais plataformas utilizadas foram PS4 e PS3, com os mesmos gêneros preferidos que na América do Norte (Action e Shooter).
# - No Japão, a plataforma mais preferida em disparado é a 3DS, com os gêneros Role-Playing e Action. 
# 
# Nas três regiões os usuários mais adquirem jogos com classificação Mature (Adulto).

# ## Testes de hipóteses

# ### Comparação das classificações dos usuários para as plataformas Xbox One e PC

# Hipóteses:
# 
# Hipótese Nula (H0): Não há diferença significativa nas classificações médias dos usuários entre as plataformas Xbox One e PC.
# Hipótese Alternativa (H1): Há uma diferença significativa nas classificações médias dos usuários entre as plataformas Xbox One e PC.


# Filtrar os dados para incluir apenas as plataformas Xbox One e PC
xbox_one_scores = games_novo[games_novo['platform'] == 'XOne']['user_score'].dropna()
pc_scores = games_novo[games_novo['platform'] == 'PC']['user_score'].dropna()

# Realizar teste de igualdade de variâncias (Bartlett's Test)
bartlett_test = stats.bartlett(xbox_one_scores, pc_scores)

# Verificar o resultado do teste de igualdade de variâncias
if bartlett_test.pvalue > 0.05:
    equal_var = True
    print("Variâncias iguais (p-value =", bartlett_test.pvalue, ")")
else:
    equal_var = False
    print("Variâncias diferentes (p-value =", bartlett_test.pvalue, ")")


# Teste t de amostras independentes
results = stats.ttest_ind(xbox_one_scores, pc_scores, equal_var=False)

print('p-value:', results.pvalue)


# Neste caso, o P-Value é 0.0188, que é menor que o valor nível de significancia de 0,05. Portanto, podemos rejeitar a hipótese nula com um nível de significância de 0,05. Isso significa que há evidências suficientes para sugerir que as médias entre as avaliações dos usuários das plataformas Xbox One e PC são diferentes.

# ### Comparação das classificações dos usuários para os gêneros Action e Sports

# Hipótese Nula (H0): Não há diferença significativa nas classificações médias dos usuários entre os gêneros Action e Sports.
# 
# Hipótese Alternativa (H1): Há uma diferença significativa nas classificações médias dos usuários entre os gêneros Action e Sports.


# Filtrar os dados para incluir apenas os gêneros Action e Sports
action_scores = games_novo[games_novo['genre'] == 'Action']['user_score'].dropna()
sports_scores = games_novo[games_novo['genre'] == 'Sports']['user_score'].dropna()

# Realizar teste de igualdade de variâncias (Bartlett's Test)
bartlett_test = stats.bartlett(action_scores, sports_scores)

# Verificar o resultado do teste de igualdade de variâncias
if bartlett_test.pvalue > 0.05:
    equal_var = True
    print("Variâncias iguais (p-value =", bartlett_test.pvalue, ")")
else:
    equal_var = False
    print("Variâncias diferentes (p-value =", bartlett_test.pvalue, ")")


# Teste t de amostras independentes
results = stats.ttest_ind(action_scores, sports_scores, equal_var=False)

print('p-value:', results.pvalue)


# Neste caso, O valor p (6.851256251893839e-12) é muito pequeno. Portanto, podemos rejeitar a hipótese nula com um nível de significância de 0,05. Isso significa que há evidências suficientes para sugerir que as médias entre as avaliações dos usuários dos gênerod Action e Sportes são diferentes.

# ## Conclusão Geral

# Com base na análise dos dados de 2016, podemos sugerir uma campanha para 2017 considerando as seguintes conclusões:
# 
# - Tendência do Mercado:
# O mercado de lançamentos de jogos atingiu seu pico em 2008-2009, estabilizando em cerca de 600 jogos por ano desde 2012.
# 
# - Plataformas Promissoras:
# As plataformas PS4, PS3 e X360 se destacaram nos últimos anos, indicando oportunidades promissoras para campanhas.
# 
# - Influência das Críticas:
# As análises profissionais são significativas para os usuários na escolha de jogos, destacando a importância de parcerias com críticos.
# 
# - Preferências de Gênero:
# Gêneros populares entre os usuários incluem Action, Shooter, Role-Playing e Sports, indicando áreas para foco de marketing.
# 
# - Diferenças Regionais:
# Estratégias de marketing devem considerar as preferências regionais, como o destaque da 3DS no Japão e as preferências de plataformas e gêneros nas Américas e Europa.
# 
# - Classificação Etária:
# Jogos com classificação Mature são preferidos em todas as regiões, sugerindo oportunidades para conteúdo direcionado a adultos.
# 
# Considerando o teste de hipótese entre as plataformas Xbox One e PC, assim como entre os gêneros Action e Sports, os resultados indicam diferenças estatisticamente significativas. Isso ressalta a importância de adaptar estratégias de marketing com base nas preferências específicas dos usuários de cada plataforma e gênero.

