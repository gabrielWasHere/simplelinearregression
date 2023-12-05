import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import sklearn
from sklearn import linear_model

# Regressão Linear SIMPLES
# Aqui lemos o arquivo e criamos um segundo arquivo contendo apenas os dados que nos interessam! (cdf)
df = pd.read_csv("FuelConsumptionCo2.csv")
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
#print(cdf.head())

# Aqui criamos uma "Máscara Booleana" que seleciona um valor aleatório para cada linha de df, sendo true para menor que 0,8
# e false para maior que 0,8. Assim temos aproximadamente uma divisão em 80%!
mascara = np.random.rand(len(df)) < 0.8
treinamento = df[mascara]
testes = df[~mascara]

# Aqui criamos um objeto do tipo Modelo de Regressão Linear.
modelo = linear_model.LinearRegression()

# Ajustamos os dados para o formato do numpy
treino_x = np.asanyarray(treinamento[['FUELCONSUMPTION_COMB']])
treino_y = np.asanyarray(treinamento[['CO2EMISSIONS']])

teste_x = np.asanyarray(testes[['FUELCONSUMPTION_COMB']])
teste_y = np.asanyarray(testes[['CO2EMISSIONS']])

# Aqui de fato treinamos o modelo
modelo.fit(treino_x, treino_y)

# Aqui testamos o modelo!
teste_result = modelo.predict(teste_x)

# Exibição de métricas

print("Erro Médio Absoluto: ", np.mean(np.absolute(teste_y - teste_result)))
print("Erro Quadrático Médio: ", np.mean((teste_y - teste_result) ** 2))
print("R Quadrado: ", sklearn.metrics.r2_score(teste_result, teste_y))

# Criação do gráfico

plt.scatter(teste_y, teste_result, marker = '+') # Gráfico de dispersão: valores reais vs previsões
# Aqui criamos a linha de "acerto", a linha y = x
plt.plot([teste_y.min(), teste_y.max()], [teste_y.min(), teste_y.max()], color='red', linestyle='--')
plt.xticks(np.arange(100, 500, 50))
plt.yticks(np.arange(100, 500, 50))
plt.xlabel('Valores Reais (Emissões)')
plt.ylabel('Valores Previstos (Emissões)')
plt.title('Valores Reais x Previsões do Modelo')
plt.show()

