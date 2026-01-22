# %%
# Manipulação de dados
import duckdb
import pandas as pd

# Tratamento de dados
from scipy import stats
import numpy as np
import gc

# Visualização
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Algoritmos
from xgboost import XGBRegressor, XGBClassifier

# Métricas de validação
from metrics import MAE,RMSE,R2_determinacao,PSC_A,PCC_A,PMC_A

# Salvar e importar modelos e métricas
import pickle
import json

# Configurações de bibliotecas
pd.set_option('display.max_columns',None)

# Importando o database de dados de modelagem
final_db = duckdb.connect(database='final_db')

# Defininindo os tipos de variáveis presentes no dataset
n_vizinhas_disponiveis = 10

list_key = ['id_estacao','dt_medicao']

list_features_geoespaciais = ['latitude','longitude','vl_declividade','vl_altitude','vl_distancia_oceano','vl_aspecto_relevo']

list_features_estacoes_temperatura = ['vl_temperatura_maxima','vl_temperatura_media','vl_temperatura_minima']
list_features_estacoes_umidade = ['vl_umidade_relativa_maxima','vl_umidade_relativa_media','vl_umidade_relativa_minima']
list_features_estacoes_vento = ['vl_velocidade_vento_2m_maxima','vl_velocidade_vento_2m_media','vl_velocidade_vento_10m_media']
list_features_estacoes = list_features_estacoes_temperatura + list_features_estacoes_umidade + list_features_estacoes_vento

list_features_chirps = ['vl_precipitacao_chirps']
list_features_cpc = ['vl_precipitacao_cpc','vl_temperatura_maxima_cpc','vl_temperatura_minima_cpc']
list_features_gpm_final_run = ['vl_precipitacao_gpm_final_run']
list_features_gpm_late_run = ['vl_precipitacao_gpm_late_run']
list_features_power = ['vl_precipitacao_power','vl_temperatura_maxima_2m_K_power','vl_temperatura_media_2m_K_power','vl_temperatura_minima_2m_K_power','vl_umidade_relativa_2m_power','vl_pressao_nivel_superficie_power','vl_irradiancia_allsky_power','vl_direcao_vento_10m_power','vl_direcao_vento_2m_power','vl_temperatura_orvalho_2m_K_power','vl_vento_10m_power','vl_vento_medio_2m_power','vl_vento_maximo_2m_power','vl_vento_maximo_10m_power']
list_features_produtos = list_features_chirps + list_features_cpc + list_features_gpm_final_run + list_features_gpm_late_run + list_features_power

list_vizinhas_aux = sum([[f'vl_correlacao_estacao_vizinha_{i}',f'pct_intersecao_precipitacao_vizinha_{i}',f'vl_distancia_km_vizinha_{i}',f'vl_prioridade_vizinha_{i}',f'vl_precipitacao_vizinha_{i}']for i in range(1, n_vizinhas_disponiveis+1)], [])
list_features_vizinhas = [f'vl_precipitacao_vizinha_{i}' for i in range(1,n_vizinhas_disponiveis+1)]

list_features_target = ['vl_precipitacao','vl_precipitacao_log']

list_split_column = ['percentil_temporal']

# %% [markdown]
# # MODELOS 1

# %%
print("""Os modelos 1 são os mais simples. Esses modelos são compostos por um único algoritmo de regressão, sempre usando XGBoost. A partição já é a padrão, de 70-30 por estação, e não há tentativas 
de modificação da target\n
> Sem pré modelo de classificação
> Sem modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Sem modificações da target
> Testando diferentes combinações de variáveis explicativas
""")

# %%
def ImportBase_Modelo1(final_db=final_db,table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()

    return df

def SplitTreinoTeste_Modelo1(df_base,pct_split,coluna_percentil_temporal='percentil_temporal'):
  df_treino = df_base.loc[df_base[coluna_percentil_temporal]<=pct_split]
  df_teste = df_base.loc[df_base[coluna_percentil_temporal]>pct_split]
  return df_treino,df_teste

def PrepararBaseTreino(df_treino,list_features,target):
   df_X_treino = df_treino.loc[:,list_features]
   df_y_treino = df_treino.loc[:,[target]]
   return df_X_treino,df_y_treino

def TreinarAlgoritmo_Modelo1(df_X_treino,df_y_treino,Algoritmo):
   alg = Algoritmo()
   alg.fit(df_X_treino,df_y_treino)
   return alg

def RealizarPredicaoTeste(df_test,list_features,target,modelo,modelo_number='1_1'):
   df_X_test = df_test[list_features]
   df_y_pred = modelo.predict(df_X_test)
   df_validacao = df_test.copy()
   df_validacao[f'{target}_modelo_{modelo_number}'] = df_y_pred
   return df_validacao

def CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva):
   metricas_modelo = {
      'MAE':MAE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'RMSE':RMSE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'R2':R2_determinacao(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'PSC_A':PSC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],psc_a_max_chuva),
      'PCC_A':PCC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pcc_a_erro),
      'PMC_A':PMC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pmc_a_erro,pmc_a_min_chuva)
   }

   return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo1(df_validacao,modelo_1_1,metricas_modelo_1_1,target,modelo_number,final_db=final_db):

    df_validacao_key_target_pred = df_validacao[list_key+[target,f'{target}_modelo_{modelo_number}']]

    final_db.execute(
    f"""
    CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
    SELECT * FROM df_validacao_key_target_pred)
    """)

    with open(f'modelos_finais/modelo_{modelo_number}.pkl','wb') as f:
        pickle.dump(modelo_1_1,f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo_1_1, f, indent=4)

    pass


# %% [markdown]
# ## Modelo 1.1

# %%
modelo_number = '1_1'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
list_features_modelo_1_1 = list_features_geoespaciais + list_features_estacoes
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo1()
df_treino,df_teste = SplitTreinoTeste_Modelo1(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_1_1,target)

# Treinando o modelo
modelo_1_1 = TreinarAlgoritmo_Modelo1(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste(df_teste,list_features_modelo_1_1,target,modelo_1_1,modelo_number)

# Calculando Métricas
metricas_modelo_1_1 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo1(df_validacao,modelo_1_1,metricas_modelo_1_1,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 1.2

# %%
modelo_number = '1_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
list_features_modelo_1_2 = list_features_geoespaciais + list_features_estacoes + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo1()
df_treino,df_teste = SplitTreinoTeste_Modelo1(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_1_2,target)

# Treinando o modelo
modelo_1_2 = TreinarAlgoritmo_Modelo1(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste(df_teste,list_features_modelo_1_2,target,modelo_1_2,modelo_number)

# Calculando Métricas
metricas_modelo_1_2 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo1(df_validacao,modelo_1_2,metricas_modelo_1_2,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 1.3

# %%
modelo_number = '1_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
list_features_modelo_1_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo1()
df_treino,df_teste = SplitTreinoTeste_Modelo1(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_1_3,target)

# Treinando o modelo
modelo_1_3 = TreinarAlgoritmo_Modelo1(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste(df_teste,list_features_modelo_1_3,target,modelo_1_3,modelo_number)

# Calculando Métricas
metricas_modelo_1_3 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo1(df_validacao,modelo_1_3,metricas_modelo_1_3,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 1.4

# %%
modelo_number = '1_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
list_features_modelo_1_4 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo1()
df_treino,df_teste = SplitTreinoTeste_Modelo1(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_1_4,target)

# Treinando o modelo
modelo_1_4 = TreinarAlgoritmo_Modelo1(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste(df_teste,list_features_modelo_1_4,target,modelo_1_4,modelo_number)

# Calculando Métricas
metricas_modelo_1_4 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo1(df_validacao,modelo_1_4,metricas_modelo_1_4,target,modelo_number,final_db=final_db)

# %% [markdown]
# # MODELOS 2

# %%
print("""Os modelos 2 são iguais aos modelos 1, porém com a variável target logarítmica. Esses modelos são compostos por um único algoritmo de regressão, sempre usando XGBoost. A partição já é a padrão, de 70-30 por estação.\n
> Sem pré modelo de classificação
> Sem modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Target Logarítmica
> Testando diferentes combinações de variáveis explicativas
""")

# %%
def ImportBase_Modelo2(final_db=final_db,table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()

    return df

def SplitTreinoTeste_Modelo2(df_base,pct_split,coluna_percentil_temporal='percentil_temporal'):
  df_treino = df_base.loc[df_base[coluna_percentil_temporal]<=pct_split]
  df_teste = df_base.loc[df_base[coluna_percentil_temporal]>pct_split]
  return df_treino,df_teste

def PrepararBaseTreino(df_treino,list_features,target):
   df_X_treino = df_treino.loc[:,list_features]
   df_y_treino = df_treino.loc[:,[target]]
   return df_X_treino,df_y_treino

def TreinarAlgoritmo_Modelo2(df_X_treino,df_y_treino,Algoritmo):
   alg = Algoritmo()
   alg.fit(df_X_treino,df_y_treino)
   return alg

def RealizarPredicaoTeste_Modelo2(df_test,list_features,target,target_original,modelo,modelo_number):
   df_X_test = df_test[list_features]
   df_y_pred = modelo.predict(df_X_test)
   df_validacao = df_test.copy()
   df_validacao[f'{target}_modelo_{modelo_number}'] = df_y_pred
   df_validacao[f'{target_original}_modelo_{modelo_number}'] = np.exp(df_validacao[f'{target}_modelo_{modelo_number}'])
   return df_validacao

def CalcularMetricasTeste_Modelo2(df_validacao,target_original,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva):
   metricas_modelo = {
      'MAE':MAE(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}']),
      'RMSE':RMSE(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}']),
      'R2':R2_determinacao(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}']),
      'PSC_A':PSC_A(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}'],psc_a_max_chuva),
      'PCC_A':PCC_A(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}'],pcc_a_erro),
      'PMC_A':PMC_A(df_validacao[target_original],df_validacao[f'{target_original}_modelo_{modelo_number}'],pmc_a_erro,pmc_a_min_chuva)
   }

   return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo2(df_validacao,modelo,metricas_modelo,target,target_original,modelo_number,final_db=final_db):

    df_validacao_key_target_pred = df_validacao[list_key+[target_original,f'{target_original}_modelo_{modelo_number}']+[target,f'{target}_modelo_{modelo_number}']]

    final_db.execute(
    f"""
    CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
    SELECT * FROM df_validacao_key_target_pred)
    """)

    with open(f'modelos_finais/modelo_{modelo_number}.pkl','wb') as f:
        pickle.dump(modelo,f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass


# %% [markdown]
# ## Modelo 2.1

# %%
modelo_number = '2_1'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao_log'
target_original = 'vl_precipitacao'
list_features_modelo_2_1 = list_features_geoespaciais + list_features_estacoes
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo2()
df_treino,df_teste = SplitTreinoTeste_Modelo2(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_2_1,target)

# Treinando o modelo
modelo_2_1 = TreinarAlgoritmo_Modelo2(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste_Modelo2(df_teste,list_features_modelo_2_1,target,target_original,modelo_2_1,modelo_number)

# Calculando Métricas
metricas_modelo_2_1 = CalcularMetricasTeste_Modelo2(df_validacao,target_original,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo2(df_validacao,modelo_2_1,metricas_modelo_2_1,target,target_original,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 2.2

# %%
modelo_number = '2_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao_log'
target_original = 'vl_precipitacao'
list_features_modelo_2_2 = list_features_geoespaciais + list_features_estacoes + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo2()
df_treino,df_teste = SplitTreinoTeste_Modelo2(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_2_2,target)

# Treinando o modelo
modelo_2_2 = TreinarAlgoritmo_Modelo2(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste_Modelo2(df_teste,list_features_modelo_2_2,target,target_original,modelo_2_2,modelo_number)

# Calculando Métricas
metricas_modelo_2_2 = CalcularMetricasTeste_Modelo2(df_validacao,target_original,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo2(df_validacao,modelo_2_2,metricas_modelo_2_2,target,target_original,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 2.3

# %%
modelo_number = '2_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao_log'
target_original = 'vl_precipitacao'
list_features_modelo_2_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo2()
df_treino,df_teste = SplitTreinoTeste_Modelo2(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_2_3,target)

# Treinando o modelo
modelo_2_3 = TreinarAlgoritmo_Modelo2(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste_Modelo2(df_teste,list_features_modelo_2_3,target,target_original,modelo_2_3,modelo_number)

# Calculando Métricas
metricas_modelo_2_3 = CalcularMetricasTeste_Modelo2(df_validacao,target_original,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo2(df_validacao,modelo_2_3,metricas_modelo_2_3,target,target_original,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 2.4

# %%
modelo_number = '2_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao_log'
target_original = 'vl_precipitacao'
list_features_modelo_2_4 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo2()
df_treino,df_teste = SplitTreinoTeste_Modelo2(df_base,pct_train_test_split)
df_X_treino,df_y_treino = PrepararBaseTreino(df_treino,list_features_modelo_2_4,target)

# Treinando o modelo
modelo_2_4 = TreinarAlgoritmo_Modelo2(df_X_treino,df_y_treino,Algoritmo)

# Validando
df_validacao = RealizarPredicaoTeste_Modelo2(df_teste,list_features_modelo_2_4,target,target_original,modelo_2_4,modelo_number)

# Calculando Métricas
metricas_modelo_2_4 = CalcularMetricasTeste_Modelo2(df_validacao,target_original,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo2(df_validacao,modelo_2_4,metricas_modelo_2_4,target,target_original,modelo_number,final_db=final_db)

# %% [markdown]
# # MODELOS 3

# %%
print("""Os modelos 3 são iguais aos modelos 1_4, porém com separação entre modelo especializado e modelo geral. Esses modelos são compostos por dois algoritmo de regressão, sempre usando XGBoost. A partição já é a padrão, de 70-30 por estação.\n
> Sem pré modelo de classificação
> Modelo especializado, testando diferentes thresholds
> Apenas algoritmo de XGBoost
> Partição padrão
> Target padrão
> Usando todas as variáveis explicativas
""")

# %%
def ImportBase_Modelo3(final_db=final_db,table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()

    return df

def SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado):
    df_base_especializado = df_base.loc[df_base[coluna_vl_prioridade_vizinha]>=threshold_modelo_especializado]
    df_base_geral = df_base.copy()
    return df_base_geral,df_base_especializado

def SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_split,coluna_percentil_temporal='percentil_temporal'):
  df_treino_especializado = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal]<=pct_split]
  df_teste_especializado = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal]>pct_split]

  df_treino_geral = df_base_geral.loc[df_base_geral[coluna_percentil_temporal]<=pct_split]
  df_teste_geral = df_base_geral.loc[df_base_geral[coluna_percentil_temporal]>pct_split]
  return df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral

def PrepararBaseTreino(df_treino,list_features,target):
   df_X_treino = df_treino.loc[:,list_features]
   df_y_treino = df_treino.loc[:,[target]]
   return df_X_treino,df_y_treino

def TreinarAlgoritmo_Modelo3(df_X_treino,df_y_treino,Algoritmo):
   alg = Algoritmo()
   alg.fit(df_X_treino,df_y_treino)
   return alg

def RealizarPredicaoTeste_Modelo3(df_test,list_features,target,modelo,modelo_number,tipo_modelo):
   df_X_test = df_test[list_features]
   df_y_pred = modelo.predict(df_X_test)
   df_validacao = df_test.copy()
   df_validacao[f'{target}_modelo_{modelo_number}_{tipo_modelo}'] = df_y_pred
   return df_validacao

def CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva):
    df_validacao_all = df_validacao_geral[['id_estacao','dt_medicao',f'vl_precipitacao_modelo_{modelo_number}_geral','vl_precipitacao']] \
        .merge(df_validacao_especializado[['id_estacao','dt_medicao',f'vl_precipitacao_modelo_{modelo_number}_especializado']],on=['id_estacao','dt_medicao'],how='left')

    df_validacao_all[f'vl_precipitacao_modelo_{modelo_number}'] = df_validacao_all[f'vl_precipitacao_modelo_{modelo_number}_especializado'].fillna(df_validacao_all[f'vl_precipitacao_modelo_{modelo_number}_geral'])
    df_validacao = df_validacao_all
    metricas_modelo = {
        'MAE':MAE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
        'RMSE':RMSE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
        'R2':R2_determinacao(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
        'PSC_A':PSC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],psc_a_max_chuva),
        'PCC_A':PCC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pcc_a_erro),
        'PMC_A':PMC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pmc_a_erro,pmc_a_min_chuva)
    }
    return metricas_modelo,df_validacao

def SalvarValidacaoModeloMetricas_Modelo3(df_validacao,modelo_geral,modelo_especializado,metricas_modelo,target,modelo_number,final_db=final_db):

    df_validacao_key_target_pred = df_validacao[list_key+[target,f'{target}_modelo_{modelo_number}']]

    final_db.execute(
    f"""
    CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
    SELECT * FROM df_validacao_key_target_pred)
    """)

    with open(f'modelos_finais/modelo_{modelo_number}_geral.pkl','wb') as f:
        pickle.dump(modelo_geral,f)

    with open(f'modelos_finais/modelo_{modelo_number}_especializado.pkl','wb') as f:
        pickle.dump(modelo_especializado,f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass


# %% [markdown]
# ## Modelo 3.1

# %%
modelo_number = '3_1'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'
list_features_modelo_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo3()

df_base_geral,df_base_especializado = SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado)
df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral = SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_train_test_split,coluna_percentil_temporal='percentil_temporal')

df_X_treino_especializado,df_y_treino_especializado = PrepararBaseTreino(df_treino_especializado,list_features_modelo_3,target)
df_X_treino_geral,df_y_treino_geral = PrepararBaseTreino(df_teste_especializado,list_features_modelo_3,target)

# Treinando os modelos especializados e geral
modelo_3_1_especializado = TreinarAlgoritmo_Modelo3(df_X_treino_especializado,df_y_treino_especializado,Algoritmo)
modelo_3_1_geral = TreinarAlgoritmo_Modelo3(df_X_treino_geral,df_y_treino_geral,Algoritmo)

# Validando
df_validacao_especializado = RealizarPredicaoTeste_Modelo3(df_teste_especializado,list_features_modelo_3,target,modelo_3_1_especializado,modelo_number,'especializado')
df_validacao_geral = RealizarPredicaoTeste_Modelo3(df_teste_geral,list_features_modelo_3,target,modelo_3_1_geral,modelo_number,'geral')

# Calculando Métricas
metricas_modelo_3_1,df_validacao_all = CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo3(df_validacao_all,modelo_3_1_geral,modelo_3_1_especializado,metricas_modelo_3_1,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 3.2

# %%
modelo_number = '3_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.75
target = 'vl_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'
list_features_modelo_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo3()

df_base_geral,df_base_especializado = SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado)
df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral = SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_train_test_split,coluna_percentil_temporal='percentil_temporal')

df_X_treino_especializado,df_y_treino_especializado = PrepararBaseTreino(df_treino_especializado,list_features_modelo_3,target)
df_X_treino_geral,df_y_treino_geral = PrepararBaseTreino(df_teste_especializado,list_features_modelo_3,target)

# Treinando os modelos especializados e geral
modelo_3_2_especializado = TreinarAlgoritmo_Modelo3(df_X_treino_especializado,df_y_treino_especializado,Algoritmo)
modelo_3_2_geral = TreinarAlgoritmo_Modelo3(df_X_treino_geral,df_y_treino_geral,Algoritmo)

# Validando
df_validacao_especializado = RealizarPredicaoTeste_Modelo3(df_teste_especializado,list_features_modelo_3,target,modelo_3_2_especializado,modelo_number,'especializado')
df_validacao_geral = RealizarPredicaoTeste_Modelo3(df_teste_geral,list_features_modelo_3,target,modelo_3_2_geral,modelo_number,'geral')

# Calculando Métricas
metricas_modelo_3_2,df_validacao_all = CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo3(df_validacao_all,modelo_3_2_geral,modelo_3_2_especializado,metricas_modelo_3_2,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 3.3

# %%
modelo_number = '3_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.9
target = 'vl_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'
list_features_modelo_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo3()

df_base_geral,df_base_especializado = SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado)
df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral = SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_train_test_split,coluna_percentil_temporal='percentil_temporal')

df_X_treino_especializado,df_y_treino_especializado = PrepararBaseTreino(df_treino_especializado,list_features_modelo_3,target)
df_X_treino_geral,df_y_treino_geral = PrepararBaseTreino(df_teste_especializado,list_features_modelo_3,target)

# Treinando os modelos especializados e geral
modelo_3_3_especializado = TreinarAlgoritmo_Modelo3(df_X_treino_especializado,df_y_treino_especializado,Algoritmo)
modelo_3_3_geral = TreinarAlgoritmo_Modelo3(df_X_treino_geral,df_y_treino_geral,Algoritmo)

# Validando
df_validacao_especializado = RealizarPredicaoTeste_Modelo3(df_teste_especializado,list_features_modelo_3,target,modelo_3_3_especializado,modelo_number,'especializado')
df_validacao_geral = RealizarPredicaoTeste_Modelo3(df_teste_geral,list_features_modelo_3,target,modelo_3_3_geral,modelo_number,'geral')

# Calculando Métricas
metricas_modelo_3_3,df_validacao_all = CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo3(df_validacao_all,modelo_3_3_geral,modelo_3_3_especializado,metricas_modelo_3_3,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 3.4

# %%
modelo_number = '3_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.4
target = 'vl_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'
list_features_modelo_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo3()

df_base_geral,df_base_especializado = SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado)
df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral = SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_train_test_split,coluna_percentil_temporal='percentil_temporal')

df_X_treino_especializado,df_y_treino_especializado = PrepararBaseTreino(df_treino_especializado,list_features_modelo_3,target)
df_X_treino_geral,df_y_treino_geral = PrepararBaseTreino(df_teste_especializado,list_features_modelo_3,target)

# Treinando os modelos especializados e geral
modelo_3_4_especializado = TreinarAlgoritmo_Modelo3(df_X_treino_especializado,df_y_treino_especializado,Algoritmo)
modelo_3_4_geral = TreinarAlgoritmo_Modelo3(df_X_treino_geral,df_y_treino_geral,Algoritmo)

# Validando
df_validacao_especializado = RealizarPredicaoTeste_Modelo3(df_teste_especializado,list_features_modelo_3,target,modelo_3_4_especializado,modelo_number,'especializado')
df_validacao_geral = RealizarPredicaoTeste_Modelo3(df_teste_geral,list_features_modelo_3,target,modelo_3_4_geral,modelo_number,'geral')

# Calculando Métricas
metricas_modelo_3_4,df_validacao_all = CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo3(df_validacao_all,modelo_3_4_geral,modelo_3_4_especializado,metricas_modelo_3_4,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 3.5

# %%
modelo_number = '3_5'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.3
target = 'vl_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'
list_features_modelo_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo3()

df_base_geral,df_base_especializado = SepararBaseEspecializadoGeral_Modelo3(df_base,coluna_vl_prioridade_vizinha,threshold_modelo_especializado)
df_treino_especializado,df_teste_especializado,df_treino_geral,df_teste_geral = SplitTreinoTeste_Modelo3(df_base_geral,df_base_especializado,pct_train_test_split,coluna_percentil_temporal='percentil_temporal')

df_X_treino_especializado,df_y_treino_especializado = PrepararBaseTreino(df_treino_especializado,list_features_modelo_3,target)
df_X_treino_geral,df_y_treino_geral = PrepararBaseTreino(df_teste_especializado,list_features_modelo_3,target)

# Treinando os modelos especializados e geral
modelo_3_5_especializado = TreinarAlgoritmo_Modelo3(df_X_treino_especializado,df_y_treino_especializado,Algoritmo)
modelo_3_5_geral = TreinarAlgoritmo_Modelo3(df_X_treino_geral,df_y_treino_geral,Algoritmo)

# Validando
df_validacao_especializado = RealizarPredicaoTeste_Modelo3(df_teste_especializado,list_features_modelo_3,target,modelo_3_5_especializado,modelo_number,'especializado')
df_validacao_geral = RealizarPredicaoTeste_Modelo3(df_teste_geral,list_features_modelo_3,target,modelo_3_5_geral,modelo_number,'geral')

# Calculando Métricas
metricas_modelo_3_5,df_validacao_all = CalcularMetricasTeste_Modelo3(df_validacao_geral,df_validacao_especializado,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo3(df_validacao_all,modelo_3_5_geral,modelo_3_5_especializado,metricas_modelo_3_5,target,modelo_number,final_db=final_db)

# %% [markdown]
# # MODELO 4

# %%
print("""Os modelos 4 são iguais aos modelos 1_4, porém com modelo de pré-classificação. Esses modelos são compostos por um algoritmo de classificação + alguns algoritmos de regressão, sempre usando XGBoost. A partição já é a padrão, de 70-30 por estação., e sem modelos especializados\n
> Com pré modelo de classificação (direto)
> Sem modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Target padrão
> Usando todas as variáveis explicativas
""")

# %%
def ImportBase_Modelo4(final_db=final_db,table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()

    return df

def SplitTreinoTeste_Modelo4(df_base,pct_split,coluna_percentil_temporal='percentil_temporal'):
  df_treino = df_base.loc[df_base[coluna_percentil_temporal]<=pct_split]
  df_teste = df_base.loc[df_base[coluna_percentil_temporal]>pct_split]
  return df_treino,df_teste

def CriarColunaClasse_Modelo4(df_base,dict_classes):
    df_base['classe_precipitacao'] = df_base['vl_precipitacao'].apply(
        lambda x: next((classe for classe, (limite_inf, limite_sup) in dict_classes.items()
                        if limite_inf <= x < limite_sup), None)
    )

    return df_base

def PrepararBaseTreino(df_treino,list_features,target):
   df_X_treino = df_treino.loc[:,list_features]
   df_y_treino = df_treino.loc[:,[target]]
   return df_X_treino,df_y_treino

def TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo):
   alg = Algoritmo()
   alg.fit(df_X_treino,df_y_treino)
   return alg

def PreverClasseTeste_Modelo4(df_teste, list_features, modelo_classificacao, nome_coluna_classe_predita='classe_predita'):
    df_teste = df_teste.copy()
    df_X_teste_classificacao = df_teste[list_features]
    df_teste[nome_coluna_classe_predita] = modelo_classificacao.predict(df_X_teste_classificacao)
    return df_teste

def RealizarPredicaoTeste_Modelo4(df_test,list_features,target,dict_modelos_regressao,modelo_number,coluna_classe_predita='classe_predita'):
    
    df_test = df_test.copy()
    dict_validacao = {}

    for classe, modelo in dict_modelos_regressao.items():
        df_teste_classe = df_test.loc[df_test[coluna_classe_predita] == classe]

        if len(df_teste_classe) == 0:
            continue

        df_X_test = df_teste_classe[list_features]
        df_y_pred = modelo.predict(df_X_test)

        df_validacao_classe = df_teste_classe.copy()
        df_validacao_classe[f'{target}_modelo_{modelo_number}'] = df_y_pred

        dict_validacao[classe] = df_validacao_classe

    if len(dict_validacao) == 0:
        return pd.DataFrame()

    df_validacao = pd.concat(dict_validacao.values(), ignore_index=True)
    return df_validacao

def CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva):
   metricas_modelo = {
      'MAE':MAE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'RMSE':RMSE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'R2':R2_determinacao(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'PSC_A':PSC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],psc_a_max_chuva),
      'PCC_A':PCC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pcc_a_erro),
      'PMC_A':PMC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pmc_a_erro,pmc_a_min_chuva)
   }

   return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_classificacao,dict_modelo_regressao,metricas_modelo,target,modelo_number,final_db=final_db):

    df_validacao_key_target_pred = df_validacao[list_key+[target,f'{target}_modelo_{modelo_number}']]

    final_db.execute(
    f"""
    CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
    SELECT * FROM df_validacao_key_target_pred)
    """)

    with open(f'modelos_finais/modelo_{modelo_number}_classificacao.pkl','wb') as f:
        pickle.dump(modelo_classificacao,f)
    
    for classe,modelo in dict_modelo_regressao.items():
       with open(f'modelos_finais/modelo_{modelo_number}_regressao_classe_{classe}.pkl','wb') as f:
          pickle.dump(modelo,f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass


# %%


# %% [markdown]
# ## Modelo 4.1

# %%
modelo_number = '4_1'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,np.inf)
}
list_features_modelo_4_1 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo4()

df_base_classes = CriarColunaClasse_Modelo4(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo4(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_4_1,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_4_1,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_4_1_classificacao = TreinarAlgoritmo_Modelo4(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_4_1_regressao = {classe:TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo4(df_teste,list_features_modelo_4_1,modelo_4_1_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo4(df_teste_com_classe,list_features_modelo_4_1,target,dict_modelo_4_1_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_4_1 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_4_1_classificacao,dict_modelo_4_1_regressao,metricas_modelo_4_1,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 4.2

# %%
modelo_number = '4_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,51), # 51 é o top 1% da base de dados
    2:(51,np.inf)
}
list_features_modelo_4_2 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo4()

df_base_classes = CriarColunaClasse_Modelo4(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo4(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_4_2,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_4_2,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_4_2_classificacao = TreinarAlgoritmo_Modelo4(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_4_2_regressao = {classe:TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo4(df_teste,list_features_modelo_4_2,modelo_4_2_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo4(df_teste_com_classe,list_features_modelo_4_2,target,dict_modelo_4_2_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_4_2 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_4_2_classificacao,dict_modelo_4_2_regressao,metricas_modelo_4_2,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 4.3

# %%
modelo_number = '4_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,23), # 23 é o top 5% da base de dados
    2:(23,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}
list_features_modelo_4_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo4()

df_base_classes = CriarColunaClasse_Modelo4(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo4(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_4_3,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_4_3,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_4_3_classificacao = TreinarAlgoritmo_Modelo4(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_4_3_regressao = {classe:TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo4(df_teste,list_features_modelo_4_3,modelo_4_3_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo4(df_teste_com_classe,list_features_modelo_4_3,target,dict_modelo_4_3_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_4_3 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_4_3_classificacao,dict_modelo_4_3_regressao,metricas_modelo_4_3,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 4.4

# %%
modelo_number = '4_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}
list_features_modelo_4_4 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo4()

df_base_classes = CriarColunaClasse_Modelo4(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo4(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_4_4,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_4_4,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_4_4_classificacao = TreinarAlgoritmo_Modelo4(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_4_4_regressao = {classe:TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo4(df_teste,list_features_modelo_4_4,modelo_4_4_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo4(df_teste_com_classe,list_features_modelo_4_4,target,dict_modelo_4_4_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_4_4 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_4_4_classificacao,dict_modelo_4_4_regressao,metricas_modelo_4_4,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 4.5

# %%
modelo_number = '4_5'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,np.inf)
}
list_features_modelo_4_5 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo4()

df_base_classes = CriarColunaClasse_Modelo4(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo4(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_4_5,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_4_5,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_4_5_classificacao = TreinarAlgoritmo_Modelo4(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_4_5_regressao = {classe:TreinarAlgoritmo_Modelo4(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo4(df_teste,list_features_modelo_4_5,modelo_4_5_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo4(df_teste_com_classe,list_features_modelo_4_5,target,dict_modelo_4_5_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_4_5 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo4(df_validacao,modelo_4_5_classificacao,dict_modelo_4_5_regressao,metricas_modelo_4_5,target,modelo_number,final_db=final_db)

# %% [markdown]
# # MODELOS 5

# %%
print("""Os modelos 5 são iguais aos modelos 4, porém com modelo de pré-classificação ponderado ao invés de direto. Esses modelos são compostos por um algoritmo de classificação + alguns algoritmos de regressão, sempre usando XGBoost. A partição já é a padrão, de 70-30 por estação., e sem modelos especializados\n
> Com pré modelo de classificação (ponderado, exceto nulo)
> Sem modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Target padrão
> Usando todas as variáveis explicativas
""")

# %%
def ImportBase_Modelo5(final_db=final_db,table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()

    return df

def SplitTreinoTeste_Modelo5(df_base,pct_split,coluna_percentil_temporal='percentil_temporal'):
  df_treino = df_base.loc[df_base[coluna_percentil_temporal]<=pct_split]
  df_teste = df_base.loc[df_base[coluna_percentil_temporal]>pct_split]
  return df_treino,df_teste

def CriarColunaClasse_Modelo5(df_base,dict_classes):
    df_base['classe_precipitacao'] = df_base['vl_precipitacao'].apply(
        lambda x: next((classe for classe, (limite_inf, limite_sup) in dict_classes.items()
                        if limite_inf <= x < limite_sup), None)
    )

    return df_base

def PrepararBaseTreino(df_treino,list_features,target):
   df_X_treino = df_treino.loc[:,list_features]
   df_y_treino = df_treino.loc[:,[target]]
   return df_X_treino,df_y_treino

def TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo):
   alg = Algoritmo()
   alg.fit(df_X_treino,df_y_treino)
   return alg

def PreverClasseTeste_Modelo5(df_teste, list_features, modelo_classificacao, nome_coluna_classe_predita='classe_predita'):
    df_teste = df_teste.copy()
    df_X_teste_classificacao = df_teste[list_features]
    
    df_teste[nome_coluna_classe_predita] = modelo_classificacao.predict(df_X_teste_classificacao)
    
    proba = modelo_classificacao.predict_proba(df_X_teste_classificacao)
    classes = modelo_classificacao.classes_
    
    for i, classe in enumerate(classes):
        df_teste[f'proba_classe_{classe}'] = proba[:, i]
    
    return df_teste

def RealizarPredicaoTeste_Modelo5(df_test, list_features, target, dict_modelos_regressao, modelo_number, coluna_classe_predita='classe_predita'):
    
    df_test = df_test.copy()
    
    df_classe_0 = df_test.loc[df_test[coluna_classe_predita] == 0]
    df_outras_classes = df_test.loc[df_test[coluna_classe_predita] != 0]
    
    list_df_validacao = []
    
    if len(df_classe_0) > 0 and 0 in dict_modelos_regressao:
        df_X_classe_0 = df_classe_0[list_features]
        df_y_pred_0 = dict_modelos_regressao[0].predict(df_X_classe_0)
        
        df_validacao_0 = df_classe_0.copy()
        df_validacao_0[f'{target}_modelo_{modelo_number}'] = df_y_pred_0
        list_df_validacao.append(df_validacao_0)
    
    if len(df_outras_classes) > 0:
        df_X_outras = df_outras_classes[list_features]
        
        predicoes_ponderadas = []
        
        for classe, modelo in dict_modelos_regressao.items():
            if classe == 0:
                continue
                
            col_proba = f'proba_classe_{classe}'
            if col_proba not in df_outras_classes.columns:
                continue
            
            y_pred_classe = modelo.predict(df_X_outras)
            
            peso = df_outras_classes[col_proba].values
            predicoes_ponderadas.append(y_pred_classe * peso)
        
        if len(predicoes_ponderadas) > 0:
            predicao_final = sum(predicoes_ponderadas)
            
            df_validacao_outras = df_outras_classes.copy()
            df_validacao_outras[f'{target}_modelo_{modelo_number}'] = predicao_final
            list_df_validacao.append(df_validacao_outras)
    
    if len(list_df_validacao) == 0:
        return pd.DataFrame()
    
    df_validacao = pd.concat(list_df_validacao, ignore_index=True)
    return df_validacao

def CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva):
   metricas_modelo = {
      'MAE':MAE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'RMSE':RMSE(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'R2':R2_determinacao(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}']),
      'PSC_A':PSC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],psc_a_max_chuva),
      'PCC_A':PCC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pcc_a_erro),
      'PMC_A':PMC_A(df_validacao[target],df_validacao[f'{target}_modelo_{modelo_number}'],pmc_a_erro,pmc_a_min_chuva)
   }

   return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_classificacao,dict_modelo_regressao,metricas_modelo,target,modelo_number,final_db=final_db):

    df_validacao_key_target_pred = df_validacao[list_key+[target,f'{target}_modelo_{modelo_number}']]

    final_db.execute(
    f"""
    CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
    SELECT * FROM df_validacao_key_target_pred)
    """)

    with open(f'modelos_finais/modelo_{modelo_number}_classificacao.pkl','wb') as f:
        pickle.dump(modelo_classificacao,f)
    
    for classe,modelo in dict_modelo_regressao.items():
       with open(f'modelos_finais/modelo_{modelo_number}_regressao_classe_{classe}.pkl','wb') as f:
          pickle.dump(modelo,f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass


# %% [markdown]
# ## Modelo 5.1

# %%
modelo_number = '5_1'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,np.inf)
}
list_features_modelo_5_1 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo5()

df_base_classes = CriarColunaClasse_Modelo5(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo5(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_5_1,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_5_1,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_5_1_classificacao = TreinarAlgoritmo_Modelo5(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_5_1_regressao = {classe:TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo5(df_teste,list_features_modelo_5_1,modelo_5_1_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo5(df_teste_com_classe,list_features_modelo_5_1,target,dict_modelo_5_1_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_5_1 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_5_1_classificacao,dict_modelo_5_1_regressao,metricas_modelo_5_1,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 5.2

# %%
modelo_number = '5_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,51), # 51 é o top 1% da base de dados
    2:(51,np.inf)
}
list_features_modelo_5_2 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo5()

df_base_classes = CriarColunaClasse_Modelo5(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo5(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_5_2,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_5_2,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_5_2_classificacao = TreinarAlgoritmo_Modelo5(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_5_2_regressao = {classe:TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo5(df_teste,list_features_modelo_5_2,modelo_5_2_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo5(df_teste_com_classe,list_features_modelo_5_2,target,dict_modelo_5_2_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_5_2 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_5_2_classificacao,dict_modelo_5_2_regressao,metricas_modelo_5_2,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 5.3

# %%
modelo_number = '5_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,23), # 23 é o top 5% da base de dados
    2:(23,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}
list_features_modelo_5_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo5()

df_base_classes = CriarColunaClasse_Modelo5(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo5(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_5_3,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_5_3,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_5_3_classificacao = TreinarAlgoritmo_Modelo5(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_5_3_regressao = {classe:TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo5(df_teste,list_features_modelo_5_3,modelo_5_3_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo5(df_teste_com_classe,list_features_modelo_5_3,target,dict_modelo_5_3_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_5_3 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_5_3_classificacao,dict_modelo_5_3_regressao,metricas_modelo_5_3,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 5.4

# %%
modelo_number = '5_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}
list_features_modelo_5_4 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo5()

df_base_classes = CriarColunaClasse_Modelo5(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo5(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_5_4,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_5_4,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_5_4_classificacao = TreinarAlgoritmo_Modelo5(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_5_4_regressao = {classe:TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo5(df_teste,list_features_modelo_5_4,modelo_5_4_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo5(df_teste_com_classe,list_features_modelo_5_4,target,dict_modelo_5_4_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_5_4 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_5_4_classificacao,dict_modelo_5_4_regressao,metricas_modelo_5_4,target,modelo_number,final_db=final_db)

# %% [markdown]
# ## Modelo 5.5

# %%
modelo_number = '5_5'

# Parâmetros de treinamento
pct_train_test_split = 0.7
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,np.inf)
}
list_features_modelo_5_5 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# Preparando base
df_base = ImportBase_Modelo5()

df_base_classes = CriarColunaClasse_Modelo5(df_base,dict_classes)
df_treino,df_teste = SplitTreinoTeste_Modelo5(df_base_classes,pct_train_test_split)

df_X_treino_classificacao,df_y_treino_classificacao = PrepararBaseTreino(df_treino,list_features_modelo_5_5,target_classificacao)

dict_treino_regressao = {classe:df_treino.loc[df_treino[target_classificacao]==classe] for classe in dict_classes.keys()}
dict_teste_regressao = {classe:df_teste.loc[df_teste[target_classificacao]==classe] for classe in dict_classes.keys()}

dict_X_y_treino_regressao = {classe:PrepararBaseTreino(df_treino_regressao,list_features_modelo_5_5,target) for classe,df_treino_regressao in dict_treino_regressao.items()}

# Treinando os modelos
modelo_5_5_classificacao = TreinarAlgoritmo_Modelo5(df_X_treino_classificacao,df_y_treino_classificacao,AlgoritmoClassificacao)
dict_modelo_5_5_regressao = {classe:TreinarAlgoritmo_Modelo5(df_X_treino,df_y_treino,Algoritmo) for classe,(df_X_treino,df_y_treino) in dict_X_y_treino_regressao.items()}

# Validando
df_teste_com_classe = PreverClasseTeste_Modelo5(df_teste,list_features_modelo_5_5,modelo_5_5_classificacao,nome_coluna_classe_predita='classe_predita')
df_validacao = RealizarPredicaoTeste_Modelo5(df_teste_com_classe,list_features_modelo_5_5,target,dict_modelo_5_5_regressao,coluna_classe_predita='classe_predita',modelo_number=modelo_number)

# Calculando Métricas
metricas_modelo_5_5 = CalcularMetricasTeste(df_validacao,target,modelo_number,psc_a_max_chuva,pcc_a_erro,pmc_a_erro,pmc_a_min_chuva)

# Salvando modelo, base de validação e métricas
SalvarValidacaoModeloMetricas_Modelo5(df_validacao,modelo_5_5_classificacao,dict_modelo_5_5_regressao,metricas_modelo_5_5,target,modelo_number,final_db=final_db)

# %% [markdown]
# # MODELOS 6

# %%
print("""Os modelos 6 combinam:
- a lógica de modelo especializado + geral dos MODELOS 3; e
- o pré-modelo de classificação dos MODELOS 4.

> Com pré modelo de classificação
> Com modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Target padrão
> Usando todas as variáveis explicativas
""")

# %%
def ImportBase_Modelo6(final_db=final_db, table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()
    return df

def CriarColunaClasse_Modelo6(df_base, dict_classes):
    """
    Igual ao CriarColunaClasse dos modelos 4/5.
    """
    df_base = df_base.copy()
    df_base['classe_precipitacao'] = df_base['vl_precipitacao'].apply(
        lambda x: next(
            (classe for classe, (limite_inf, limite_sup) in dict_classes.items()
             if limite_inf <= x < limite_sup),
            None
        )
    )
    return df_base

def SepararBaseEspecializadoGeral_Modelo6(df_base, coluna_vl_prioridade_vizinha, threshold_modelo_especializado):
    """
    Igual à lógica do Modelo3: separa base geral e especializada.
    """
    df_base_especializado = df_base.loc[df_base[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado]
    df_base_geral = df_base.copy()
    return df_base_geral, df_base_especializado

def SplitTreinoTeste_Modelo6(df_base_geral, df_base_especializado, pct_split, coluna_percentil_temporal='percentil_temporal'):
    """
    Split temporal para geral e especializado (como no Modelo 3).
    """
    df_treino_especializado = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal] <= pct_split]
    df_teste_especializado  = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal] >  pct_split]

    df_treino_geral = df_base_geral.loc[df_base_geral[coluna_percentil_temporal] <= pct_split]
    df_teste_geral  = df_base_geral.loc[df_base_geral[coluna_percentil_temporal] >  pct_split]

    return df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral

def PrepararBaseTreino(df_treino, list_features, target):
    df_X_treino = df_treino.loc[:, list_features]
    df_y_treino = df_treino.loc[:, [target]]
    return df_X_treino, df_y_treino

def TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo):
    alg = Algoritmo()
    alg.fit(df_X_treino, df_y_treino)
    return alg

def TreinarClassificacao_Modelo6(df_X_treino_classificacao, df_y_treino_classificacao, AlgoritmoClassificacao):
    modelo_classificacao = AlgoritmoClassificacao()
    modelo_classificacao.fit(df_X_treino_classificacao, df_y_treino_classificacao)
    return modelo_classificacao

def PreverClasseTeste_Modelo6(df_teste, list_features, modelo_classificacao, nome_coluna_classe_predita='classe_predita'):
    """
    Igual ao PreverClasseTeste_Modelo4, mas reaproveitado aqui.
    """
    df_teste = df_teste.copy()
    df_X_teste_classificacao = df_teste[list_features]
    df_teste[nome_coluna_classe_predita] = modelo_classificacao.predict(df_X_teste_classificacao)
    return df_teste

def RealizarPredicaoTeste_Modelo6(df_test,
                                  list_features,
                                  target,
                                  dict_modelos_regressao_geral,
                                  dict_modelos_regressao_especializado,
                                  modelo_number,
                                  coluna_classe_predita='classe_predita',
                                  coluna_flag_especializado='flag_especializado'):
    """
    Para cada registro:
      - se flag_especializado == 1 e existir modelo especializado para a classe -> usa especializado
      - senão -> usa o modelo geral da classe.
    A estrutura de dicts é: {classe: modelo}.
    """
    df_test = df_test.copy()
    dict_validacao = {}

    for classe in dict_modelos_regressao_geral.keys():
        df_teste_classe = df_test.loc[df_test[coluna_classe_predita] == classe]

        if len(df_teste_classe) == 0:
            continue

        df_teste_classe_esp = df_teste_classe.loc[df_teste_classe[coluna_flag_especializado] == 1]
        df_teste_classe_ger = df_teste_classe.loc[df_teste_classe[coluna_flag_especializado] != 1]

        list_df_val_classe = []

        if classe in dict_modelos_regressao_especializado and len(df_teste_classe_esp) > 0:
            df_X_esp = df_teste_classe_esp[list_features]
            modelo_esp = dict_modelos_regressao_especializado[classe]
            y_pred_esp = modelo_esp.predict(df_X_esp)

            df_val_esp = df_teste_classe_esp.copy()
            df_val_esp[f'{target}_modelo_{modelo_number}'] = y_pred_esp
            list_df_val_classe.append(df_val_esp)

        if len(df_teste_classe_ger) > 0:
            df_X_ger = df_teste_classe_ger[list_features]
            modelo_geral = dict_modelos_regressao_geral[classe]
            y_pred_ger = modelo_geral.predict(df_X_ger)

            df_val_ger = df_teste_classe_ger.copy()
            df_val_ger[f'{target}_modelo_{modelo_number}'] = y_pred_ger
            list_df_val_classe.append(df_val_ger)

        if len(list_df_val_classe) > 0:
            df_validacao_classe = pd.concat(list_df_val_classe, ignore_index=True)
            dict_validacao[classe] = df_validacao_classe

    if len(dict_validacao) == 0:
        return pd.DataFrame()

    df_validacao = pd.concat(dict_validacao.values(), ignore_index=True)
    return df_validacao

def CalcularMetricasTeste_Modelo6(df_validacao, target, modelo_number,
                                  psc_a_max_chuva, pcc_a_erro, pmc_a_erro, pmc_a_min_chuva):
    metricas_modelo = {
        'MAE':   MAE(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'RMSE':  RMSE(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'R2':    R2_determinacao(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'PSC_A': PSC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], psc_a_max_chuva),
        'PCC_A': PCC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], pcc_a_erro),
        'PMC_A': PMC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], pmc_a_erro, pmc_a_min_chuva)
    }
    return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo6(df_validacao,
                                          modelo_classificacao,
                                          dict_modelos_regressao_geral,
                                          dict_modelos_regressao_especializado,
                                          metricas_modelo,
                                          target,
                                          modelo_number,
                                          final_db=final_db):
    df_validacao_key_target_pred = df_validacao[list_key + [target, f'{target}_modelo_{modelo_number}']]

    final_db.execute(
        f"""
        CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
            SELECT * FROM df_validacao_key_target_pred
        )
        """
    )

    with open(f'modelos_finais/modelo_{modelo_number}_classificacao.pkl', 'wb') as f:
        pickle.dump(modelo_classificacao, f)

    for classe, modelo in dict_modelos_regressao_geral.items():
        with open(f'modelos_finais/modelo_{modelo_number}_regressao_geral_classe_{classe}.pkl', 'wb') as f:
            pickle.dump(modelo, f)

    for classe, modelo in dict_modelos_regressao_especializado.items():
        with open(f'modelos_finais/modelo_{modelo_number}_regressao_especializado_classe_{classe}.pkl', 'wb') as f:
            pickle.dump(modelo, f)

    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass

# %% [markdown]
# ## Modelo 6.1

# %%
modelo_number = '6_1'

# Parâmetros de treinamento (mesmos de 3_1 + 4_1)
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5               # igual 3_1
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

# Classes iguais ao 4_1
dict_classes = {
    0: (0, 1),
    1: (1, np.inf)
}

list_features_modelo_6_1 = (
    list_features_geoespaciais
    + list_features_estacoes
    + list_features_vizinhas
    + list_features_produtos
)
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# =====================================================================
# Preparando base
# =====================================================================
df_base = ImportBase_Modelo6()

df_base_classes = CriarColunaClasse_Modelo6(df_base, dict_classes)

# df_base não é mais usado depois daqui
del df_base

df_base_classes['flag_especializado'] = (
    df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado
).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo6(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo6(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# df_base_geral e df_base_especializado não são mais usados
del df_base_geral, df_base_especializado

# =====================================================================
# Treino do modelo de classificação (usando df_treino_geral com todas as classes)
# =====================================================================
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_6_1,
    target_classificacao
)

modelo_6_1_classificacao = TreinarClassificacao_Modelo6(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Features de treino de classificação não são mais utilizados
del df_X_treino_classificacao, df_y_treino_classificacao

# =====================================================================
# Treino dos modelos de regressão por classe - geral e especializado
# =====================================================================
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_1, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_1, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

# df_treino_geral / df_treino_especializado e dict_treino_* não são mais usados
del df_treino_geral, df_treino_especializado
del dict_treino_regressao_geral, dict_treino_regressao_especializado

dict_modelo_6_1_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_6_1_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# X e y de treino de regressão não são mais usados
del dict_X_y_treino_regressao_geral, dict_X_y_treino_regressao_especializado

# =====================================================================
# Preparação do conjunto de teste
# =====================================================================
df_teste_all = pd.concat(
    [df_teste_geral, df_teste_especializado],
    ignore_index=True
).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais usados
del df_teste_geral, df_teste_especializado

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# df_base_classes só serve para o merge acima. Depois disso não é mais usado.
del df_base_classes

df_teste_com_classe_pred = PreverClasseTeste_Modelo6(
    df_teste_all,
    list_features_modelo_6_1,
    modelo_6_1_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não é mais necessário
del df_teste_all

df_validacao_6_1 = RealizarPredicaoTeste_Modelo6(
    df_teste_com_classe_pred,
    list_features_modelo_6_1,
    target,
    dict_modelo_6_1_regressao_geral,
    dict_modelo_6_1_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred

metricas_modelo_6_1 = CalcularMetricasTeste_Modelo6(
    df_validacao_6_1,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

SalvarValidacaoModeloMetricas_Modelo6(
    df_validacao_6_1,
    modelo_6_1_classificacao,
    dict_modelo_6_1_regressao_geral,
    dict_modelo_6_1_regressao_especializado,
    metricas_modelo_6_1,
    target,
    modelo_number,
    final_db=final_db
)

# Se o script termina aqui, o GC vai limpar de qualquer forma.
# Mas se for um notebook longo, você ainda pode:
# del df_validacao_6_1, metricas_modelo_6_1

# %% [markdown]
# ## Modelo 6.2

# %%
modelo_number = '6_2'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,51), # 51 é o top 1% da base de dados
    2:(51,np.inf)
}

list_features_modelo_6_2 = (
    list_features_geoespaciais
    + list_features_estacoes
    + list_features_vizinhas
    + list_features_produtos
)
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# =====================================================================
# Preparando base
# =====================================================================
df_base = ImportBase_Modelo6()

df_base_classes = CriarColunaClasse_Modelo6(df_base, dict_classes)

del df_base

df_base_classes['flag_especializado'] = (
    df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado
).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo6(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo6(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

del df_base_geral, df_base_especializado

# =====================================================================
# Treino do modelo de classificação (usando df_treino_geral com todas as classes)
# =====================================================================
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_6_2,
    target_classificacao
)

modelo_6_2_classificacao = TreinarClassificacao_Modelo6(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

del df_X_treino_classificacao, df_y_treino_classificacao

# =====================================================================
# Treino dos modelos de regressão por classe - geral e especializado
# =====================================================================
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_2, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_2, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

del df_treino_geral, df_treino_especializado
del dict_treino_regressao_geral, dict_treino_regressao_especializado

dict_modelo_6_2_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_6_2_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

del dict_X_y_treino_regressao_geral, dict_X_y_treino_regressao_especializado

# =====================================================================
# Preparação do conjunto de teste
# =====================================================================
df_teste_all = pd.concat(
    [df_teste_geral, df_teste_especializado],
    ignore_index=True
).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

del df_teste_geral, df_teste_especializado

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

del df_base_classes

df_teste_com_classe_pred = PreverClasseTeste_Modelo6(
    df_teste_all,
    list_features_modelo_6_2,
    modelo_6_2_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

del df_teste_all

df_validacao_6_2 = RealizarPredicaoTeste_Modelo6(
    df_teste_com_classe_pred,
    list_features_modelo_6_2,
    target,
    dict_modelo_6_2_regressao_geral,
    dict_modelo_6_2_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

del df_teste_com_classe_pred

metricas_modelo_6_2 = CalcularMetricasTeste_Modelo6(
    df_validacao_6_2,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

SalvarValidacaoModeloMetricas_Modelo6(
    df_validacao_6_2,
    modelo_6_2_classificacao,
    dict_modelo_6_2_regressao_geral,
    dict_modelo_6_2_regressao_especializado,
    metricas_modelo_6_2,
    target,
    modelo_number,
    final_db=final_db
)

# Se o script termina aqui, o GC vai limpar de qualquer forma.
# Mas se for um notebook longo, você ainda pode:
# del df_validacao_6_2, metricas_modelo_6_2

# %% [markdown]
# ## Modelo 6.3

# %%
modelo_number = '6_3'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,23), # 23 é o top 5% da base de dados
    2:(23,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}

list_features_modelo_6_3 = (
    list_features_geoespaciais
    + list_features_estacoes
    + list_features_vizinhas
    + list_features_produtos
)
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# =====================================================================
# Preparando base
# =====================================================================
df_base = ImportBase_Modelo6()

df_base_classes = CriarColunaClasse_Modelo6(df_base, dict_classes)

del df_base

df_base_classes['flag_especializado'] = (
    df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado
).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo6(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo6(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

del df_base_geral, df_base_especializado

# =====================================================================
# Treino do modelo de classificação (usando df_treino_geral com todas as classes)
# =====================================================================
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_6_3,
    target_classificacao
)

modelo_6_3_classificacao = TreinarClassificacao_Modelo6(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

del df_X_treino_classificacao, df_y_treino_classificacao

# =====================================================================
# Treino dos modelos de regressão por classe - geral e especializado
# =====================================================================
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_3, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_3, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

del df_treino_geral, df_treino_especializado
del dict_treino_regressao_geral, dict_treino_regressao_especializado

dict_modelo_6_3_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_6_3_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

del dict_X_y_treino_regressao_geral, dict_X_y_treino_regressao_especializado

# =====================================================================
# Preparação do conjunto de teste
# =====================================================================
df_teste_all = pd.concat(
    [df_teste_geral, df_teste_especializado],
    ignore_index=True
).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

del df_teste_geral, df_teste_especializado

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

del df_base_classes

df_teste_com_classe_pred = PreverClasseTeste_Modelo6(
    df_teste_all,
    list_features_modelo_6_3,
    modelo_6_3_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

del df_teste_all

df_validacao_6_3 = RealizarPredicaoTeste_Modelo6(
    df_teste_com_classe_pred,
    list_features_modelo_6_3,
    target,
    dict_modelo_6_3_regressao_geral,
    dict_modelo_6_3_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

del df_teste_com_classe_pred

metricas_modelo_6_3 = CalcularMetricasTeste_Modelo6(
    df_validacao_6_3,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

SalvarValidacaoModeloMetricas_Modelo6(
    df_validacao_6_3,
    modelo_6_3_classificacao,
    dict_modelo_6_3_regressao_geral,
    dict_modelo_6_3_regressao_especializado,
    metricas_modelo_6_3,
    target,
    modelo_number,
    final_db=final_db
)

# Se o script termina aqui, o GC vai limpar de qualquer forma.
# Mas se for um notebook longo, você ainda pode:
# del df_validacao_6_3, metricas_modelo_6_3

# %% [markdown]
# ## Modelo 6.4

# %%
modelo_number = '6_4'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}

list_features_modelo_6_4 = (
    list_features_geoespaciais
    + list_features_estacoes
    + list_features_vizinhas
    + list_features_produtos
)
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# =====================================================================
# Preparando base
# =====================================================================
df_base = ImportBase_Modelo6()

df_base_classes = CriarColunaClasse_Modelo6(df_base, dict_classes)

del df_base

df_base_classes['flag_especializado'] = (
    df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado
).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo6(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo6(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

del df_base_geral, df_base_especializado

# =====================================================================
# Treino do modelo de classificação (usando df_treino_geral com todas as classes)
# =====================================================================
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_6_4,
    target_classificacao
)

modelo_6_4_classificacao = TreinarClassificacao_Modelo6(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

del df_X_treino_classificacao, df_y_treino_classificacao

# =====================================================================
# Treino dos modelos de regressão por classe - geral e especializado
# =====================================================================
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_4, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_4, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

del df_treino_geral, df_treino_especializado
del dict_treino_regressao_geral, dict_treino_regressao_especializado

dict_modelo_6_4_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_6_4_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

del dict_X_y_treino_regressao_geral, dict_X_y_treino_regressao_especializado

# =====================================================================
# Preparação do conjunto de teste
# =====================================================================
df_teste_all = pd.concat(
    [df_teste_geral, df_teste_especializado],
    ignore_index=True
).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

del df_teste_geral, df_teste_especializado

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

del df_base_classes

df_teste_com_classe_pred = PreverClasseTeste_Modelo6(
    df_teste_all,
    list_features_modelo_6_4,
    modelo_6_4_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

del df_teste_all

df_validacao_6_4 = RealizarPredicaoTeste_Modelo6(
    df_teste_com_classe_pred,
    list_features_modelo_6_4,
    target,
    dict_modelo_6_4_regressao_geral,
    dict_modelo_6_4_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

del df_teste_com_classe_pred

metricas_modelo_6_4 = CalcularMetricasTeste_Modelo6(
    df_validacao_6_4,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

SalvarValidacaoModeloMetricas_Modelo6(
    df_validacao_6_4,
    modelo_6_4_classificacao,
    dict_modelo_6_4_regressao_geral,
    dict_modelo_6_4_regressao_especializado,
    metricas_modelo_6_4,
    target,
    modelo_number,
    final_db=final_db
)

# Se o script termina aqui, o GC vai limpar de qualquer forma.
# Mas se for um notebook longo, você ainda pode:
# del df_validacao_6_4, metricas_modelo_6_4

# %% [markdown]
# ## Modelo 6.5

# %%
modelo_number = '6_5'

# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,np.inf)
}

list_features_modelo_6_5 = (
    list_features_geoespaciais
    + list_features_estacoes
    + list_features_vizinhas
    + list_features_produtos
)
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

# =====================================================================
# Preparando base
# =====================================================================
df_base = ImportBase_Modelo6()

df_base_classes = CriarColunaClasse_Modelo6(df_base, dict_classes)

del df_base

df_base_classes['flag_especializado'] = (
    df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado
).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo6(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo6(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

del df_base_geral, df_base_especializado

# =====================================================================
# Treino do modelo de classificação (usando df_treino_geral com todas as classes)
# =====================================================================
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_6_5,
    target_classificacao
)

modelo_6_5_classificacao = TreinarClassificacao_Modelo6(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

del df_X_treino_classificacao, df_y_treino_classificacao

# =====================================================================
# Treino dos modelos de regressão por classe - geral e especializado
# =====================================================================
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_5, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_6_5, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

del df_treino_geral, df_treino_especializado
del dict_treino_regressao_geral, dict_treino_regressao_especializado

dict_modelo_6_5_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_6_5_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo6(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

del dict_X_y_treino_regressao_geral, dict_X_y_treino_regressao_especializado

# =====================================================================
# Preparação do conjunto de teste
# =====================================================================
df_teste_all = pd.concat(
    [df_teste_geral, df_teste_especializado],
    ignore_index=True
).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

del df_teste_geral, df_teste_especializado

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

del df_base_classes

df_teste_com_classe_pred = PreverClasseTeste_Modelo6(
    df_teste_all,
    list_features_modelo_6_5,
    modelo_6_5_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

del df_teste_all

df_validacao_6_5 = RealizarPredicaoTeste_Modelo6(
    df_teste_com_classe_pred,
    list_features_modelo_6_5,
    target,
    dict_modelo_6_5_regressao_geral,
    dict_modelo_6_5_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

del df_teste_com_classe_pred

metricas_modelo_6_5 = CalcularMetricasTeste_Modelo6(
    df_validacao_6_5,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

SalvarValidacaoModeloMetricas_Modelo6(
    df_validacao_6_5,
    modelo_6_5_classificacao,
    dict_modelo_6_5_regressao_geral,
    dict_modelo_6_5_regressao_especializado,
    metricas_modelo_6_5,
    target,
    modelo_number,
    final_db=final_db
)

# Se o script termina aqui, o GC vai limpar de qualquer forma.
# Mas se for um notebook longo, você ainda pode:
# del df_validacao_6_5, metricas_modelo_6_5

# %% [markdown]
# # MODELOS 7

# %%
print("""Os modelos 7 combinam:
- a lógica de modelo especializado + geral dos MODELOS 3; e
- o pré-modelo de classificação PONDERADO dos MODELOS 5.

> Com pré modelo de classificação (ponderado, exceto nulo)
> Com modelo especializado
> Apenas algoritmo de XGBoost
> Partição padrão
> Target padrão
> Usando todas as variáveis explicativas
""")

# %%
def ImportBase_Modelo7(final_db=final_db, table_name='abt_base'):
    df = final_db.execute(f"""
    SELECT
        *
    FROM {table_name}
    """).fetch_df()
    return df

def CriarColunaClasse_Modelo7(df_base, dict_classes):
    """
    Cria coluna de classe de precipitação.
    """
    df_base = df_base.copy()
    df_base['classe_precipitacao'] = df_base['vl_precipitacao'].apply(
        lambda x: next(
            (classe for classe, (limite_inf, limite_sup) in dict_classes.items()
             if limite_inf <= x < limite_sup),
            None
        )
    )
    return df_base

def SepararBaseEspecializadoGeral_Modelo7(df_base, coluna_vl_prioridade_vizinha, threshold_modelo_especializado):
    """
    Separa base geral e especializada baseado no threshold de prioridade.
    """
    df_base_especializado = df_base.loc[df_base[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado]
    df_base_geral = df_base.copy()
    return df_base_geral, df_base_especializado

def SplitTreinoTeste_Modelo7(df_base_geral, df_base_especializado, pct_split, coluna_percentil_temporal='percentil_temporal'):
    """
    Split temporal para geral e especializado.
    """
    df_treino_especializado = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal] <= pct_split]
    df_teste_especializado  = df_base_especializado.loc[df_base_especializado[coluna_percentil_temporal] >  pct_split]

    df_treino_geral = df_base_geral.loc[df_base_geral[coluna_percentil_temporal] <= pct_split]
    df_teste_geral  = df_base_geral.loc[df_base_geral[coluna_percentil_temporal] >  pct_split]

    return df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral

def PrepararBaseTreino(df_treino, list_features, target):
    df_X_treino = df_treino.loc[:, list_features]
    df_y_treino = df_treino.loc[:, [target]]
    return df_X_treino, df_y_treino

def TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo):
    alg = Algoritmo()
    alg.fit(df_X_treino, df_y_treino)
    return alg

def TreinarClassificacao_Modelo7(df_X_treino_classificacao, df_y_treino_classificacao, AlgoritmoClassificacao):
    modelo_classificacao = AlgoritmoClassificacao()
    modelo_classificacao.fit(df_X_treino_classificacao, df_y_treino_classificacao)
    return modelo_classificacao

def PreverClasseTeste_Modelo7(df_teste, list_features, modelo_classificacao, nome_coluna_classe_predita='classe_predita'):
    """
    Prediz a classe e também as probabilidades de cada classe (para ponderação).
    """
    df_teste = df_teste.copy()
    df_X_teste_classificacao = df_teste[list_features]
    
    df_teste[nome_coluna_classe_predita] = modelo_classificacao.predict(df_X_teste_classificacao)
    
    # Probabilidades para ponderação
    proba = modelo_classificacao.predict_proba(df_X_teste_classificacao)
    classes = modelo_classificacao.classes_
    
    for i, classe in enumerate(classes):
        df_teste[f'proba_classe_{classe}'] = proba[:, i]
    
    return df_teste

def RealizarPredicaoTeste_Modelo7(df_test,
                                  list_features,
                                  target,
                                  dict_modelos_regressao_geral,
                                  dict_modelos_regressao_especializado,
                                  modelo_number,
                                  coluna_classe_predita='classe_predita',
                                  coluna_flag_especializado='flag_especializado'):
    """
    Predição ponderada (exceto classe 0) combinando especializado + geral.
    
    Lógica:
    - Classe 0: predição direta (sem ponderação)
    - Outras classes: ponderação por probabilidade
    - Para cada registro, usa modelo especializado se flag_especializado==1, senão usa geral
    """
    df_test = df_test.copy()
    
    # Separar classe 0 e outras classes
    df_classe_0 = df_test.loc[df_test[coluna_classe_predita] == 0]
    df_outras_classes = df_test.loc[df_test[coluna_classe_predita] != 0]
    
    list_df_validacao = []
    
    # ===== CLASSE 0: Predição direta (sem ponderação) =====
    if len(df_classe_0) > 0 and 0 in dict_modelos_regressao_geral:
        # Separar especializado e geral dentro da classe 0
        df_classe_0_esp = df_classe_0.loc[df_classe_0[coluna_flag_especializado] == 1]
        df_classe_0_ger = df_classe_0.loc[df_classe_0[coluna_flag_especializado] != 1]
        
        # Especializado
        if 0 in dict_modelos_regressao_especializado and len(df_classe_0_esp) > 0:
            df_X_esp = df_classe_0_esp[list_features]
            y_pred_esp = dict_modelos_regressao_especializado[0].predict(df_X_esp)
            
            df_val_esp = df_classe_0_esp.copy()
            df_val_esp[f'{target}_modelo_{modelo_number}'] = y_pred_esp
            list_df_validacao.append(df_val_esp)
        
        # Geral
        if len(df_classe_0_ger) > 0:
            df_X_ger = df_classe_0_ger[list_features]
            y_pred_ger = dict_modelos_regressao_geral[0].predict(df_X_ger)
            
            df_val_ger = df_classe_0_ger.copy()
            df_val_ger[f'{target}_modelo_{modelo_number}'] = y_pred_ger
            list_df_validacao.append(df_val_ger)
    
    # ===== OUTRAS CLASSES: Predição ponderada =====
    if len(df_outras_classes) > 0:
        # Separar especializado e geral
        df_outras_esp = df_outras_classes.loc[df_outras_classes[coluna_flag_especializado] == 1]
        df_outras_ger = df_outras_classes.loc[df_outras_classes[coluna_flag_especializado] != 1]
        
        # Processar especializado
        if len(df_outras_esp) > 0:
            df_X_esp = df_outras_esp[list_features]
            predicoes_ponderadas_esp = []
            
            for classe, modelo in dict_modelos_regressao_especializado.items():
                if classe == 0:
                    continue
                
                col_proba = f'proba_classe_{classe}'
                if col_proba not in df_outras_esp.columns:
                    continue
                
                y_pred_classe = modelo.predict(df_X_esp)
                peso = df_outras_esp[col_proba].values
                predicoes_ponderadas_esp.append(y_pred_classe * peso)
            
            if len(predicoes_ponderadas_esp) > 0:
                predicao_final_esp = sum(predicoes_ponderadas_esp)
                
                df_val_esp = df_outras_esp.copy()
                df_val_esp[f'{target}_modelo_{modelo_number}'] = predicao_final_esp
                list_df_validacao.append(df_val_esp)
        
        # Processar geral
        if len(df_outras_ger) > 0:
            df_X_ger = df_outras_ger[list_features]
            predicoes_ponderadas_ger = []
            
            for classe, modelo in dict_modelos_regressao_geral.items():
                if classe == 0:
                    continue
                
                col_proba = f'proba_classe_{classe}'
                if col_proba not in df_outras_ger.columns:
                    continue
                
                y_pred_classe = modelo.predict(df_X_ger)
                peso = df_outras_ger[col_proba].values
                predicoes_ponderadas_ger.append(y_pred_classe * peso)
            
            if len(predicoes_ponderadas_ger) > 0:
                predicao_final_ger = sum(predicoes_ponderadas_ger)
                
                df_val_ger = df_outras_ger.copy()
                df_val_ger[f'{target}_modelo_{modelo_number}'] = predicao_final_ger
                list_df_validacao.append(df_val_ger)
    
    if len(list_df_validacao) == 0:
        return pd.DataFrame()
    
    df_validacao = pd.concat(list_df_validacao, ignore_index=True)
    return df_validacao

def CalcularMetricasTeste_Modelo7(df_validacao, target, modelo_number,
                                  psc_a_max_chuva, pcc_a_erro, pmc_a_erro, pmc_a_min_chuva):
    metricas_modelo = {
        'MAE':   MAE(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'RMSE':  RMSE(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'R2':    R2_determinacao(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}']),
        'PSC_A': PSC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], psc_a_max_chuva),
        'PCC_A': PCC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], pcc_a_erro),
        'PMC_A': PMC_A(df_validacao[target], df_validacao[f'{target}_modelo_{modelo_number}'], pmc_a_erro, pmc_a_min_chuva)
    }
    return metricas_modelo

def SalvarValidacaoModeloMetricas_Modelo7(df_validacao,
                                          modelo_classificacao,
                                          dict_modelos_regressao_geral,
                                          dict_modelos_regressao_especializado,
                                          metricas_modelo,
                                          target,
                                          modelo_number,
                                          final_db=final_db):
    df_validacao_key_target_pred = df_validacao[list_key + [target, f'{target}_modelo_{modelo_number}']]

    final_db.execute(
        f"""
        CREATE OR REPLACE TABLE tb_validacao_modelo_{modelo_number} AS (
            SELECT * FROM df_validacao_key_target_pred
        )
        """
    )

    # Salvar modelo de classificação
    with open(f'modelos_finais/modelo_{modelo_number}_classificacao.pkl', 'wb') as f:
        pickle.dump(modelo_classificacao, f)

    # Salvar modelos gerais por classe
    for classe, modelo in dict_modelos_regressao_geral.items():
        with open(f'modelos_finais/modelo_{modelo_number}_regressao_geral_classe_{classe}.pkl', 'wb') as f:
            pickle.dump(modelo, f)

    # Salvar modelos especializados por classe
    for classe, modelo in dict_modelos_regressao_especializado.items():
        with open(f'modelos_finais/modelo_{modelo_number}_regressao_especializado_classe_{classe}.pkl', 'wb') as f:
            pickle.dump(modelo, f)

    # Salvar métricas
    with open(f'modelos_finais/metricas_{modelo_number}.json', 'w') as f:
        json.dump(metricas_modelo, f, indent=4)

    pass

# %% [markdown]
# ## Modelo 7.1

# %%
# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0: (0, 1),
    1: (1, np.inf)
}

list_features_modelo_7_1 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

modelo_number = '7_1'

# ===== Carregando base =====
df_base = ImportBase_Modelo7()

df_base_classes = CriarColunaClasse_Modelo7(df_base, dict_classes)

df_base_classes['flag_especializado'] = (df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo7(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo7(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# Já podemos descartar as bases completas
del df_base, df_base_classes, df_base_geral, df_base_especializado
gc.collect()

# ===== Treino do modelo de classificação (usando df_treino_geral com todas as classes) =====
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_7_1,
    target_classificacao
)

modelo_7_1_classificacao = TreinarClassificacao_Modelo7(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Libera X e y de classificação (não serão mais usados)
del df_X_treino_classificacao, df_y_treino_classificacao
gc.collect()

# ===== Treino dos modelos de regressão por classe - geral e especializado =====
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_1, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_1, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

dict_modelo_7_1_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_7_1_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# Podemos liberar tudo que era de treino
del (
    df_treino_geral,
    df_treino_especializado,
    dict_treino_regressao_geral,
    dict_treino_regressao_especializado,
    dict_X_y_treino_regressao_geral,
    dict_X_y_treino_regressao_especializado
)
gc.collect()

# ===== Validação =====
df_teste_all = pd.concat([df_teste_geral, df_teste_especializado], ignore_index=True).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais necessários depois de df_teste_all
del df_teste_geral, df_teste_especializado
gc.collect()

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# OBS: se tiver removido df_base_classes lá em cima, remova também este merge
# ou adapte para carregar flag_especializado de outra fonte. Caso mantenha,
# não dê del em df_base_classes antes deste ponto.

df_teste_com_classe_pred = PreverClasseTeste_Modelo7(
    df_teste_all,
    list_features_modelo_7_1,
    modelo_7_1_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não será mais usado
del df_teste_all
gc.collect()

df_validacao_7_1 = RealizarPredicaoTeste_Modelo7(
    df_teste_com_classe_pred,
    list_features_modelo_7_1,
    target,
    dict_modelo_7_1_regressao_geral,
    dict_modelo_7_1_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred
gc.collect()

# ===== Métricas =====
metricas_modelo_7_1 = CalcularMetricasTeste_Modelo7(
    df_validacao_7_1,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

# ===== Salvando =====
SalvarValidacaoModeloMetricas_Modelo7(
    df_validacao_7_1,
    modelo_7_1_classificacao,
    dict_modelo_7_1_regressao_geral,
    dict_modelo_7_1_regressao_especializado,
    metricas_modelo_7_1,
    target,
    modelo_number,
    final_db=final_db
)

# Após salvar, podemos liberar praticamente tudo
del (
    df_validacao_7_1,
    modelo_7_1_classificacao,
    dict_modelo_7_1_regressao_geral,
    dict_modelo_7_1_regressao_especializado,
    metricas_modelo_7_1
)
gc.collect()

# %% [markdown]
# ## Modelo 7.2

# %%
# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,51), # 51 é o top 1% da base de dados
    2:(51,np.inf)
}

list_features_modelo_7_2 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

modelo_number = '7_2'

# ===== Carregando base =====
df_base = ImportBase_Modelo7()

df_base_classes = CriarColunaClasse_Modelo7(df_base, dict_classes)

df_base_classes['flag_especializado'] = (df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo7(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo7(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# Já podemos descartar as bases completas
del df_base, df_base_classes, df_base_geral, df_base_especializado
gc.collect()

# ===== Treino do modelo de classificação (usando df_treino_geral com todas as classes) =====
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_7_2,
    target_classificacao
)

modelo_7_2_classificacao = TreinarClassificacao_Modelo7(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Libera X e y de classificação (não serão mais usados)
del df_X_treino_classificacao, df_y_treino_classificacao
gc.collect()

# ===== Treino dos modelos de regressão por classe - geral e especializado =====
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_2, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_2, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

dict_modelo_7_2_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_7_2_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# Podemos liberar tudo que era de treino
del (
    df_treino_geral,
    df_treino_especializado,
    dict_treino_regressao_geral,
    dict_treino_regressao_especializado,
    dict_X_y_treino_regressao_geral,
    dict_X_y_treino_regressao_especializado
)
gc.collect()

# ===== Validação =====
df_teste_all = pd.concat([df_teste_geral, df_teste_especializado], ignore_index=True).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais necessários depois de df_teste_all
del df_teste_geral, df_teste_especializado
gc.collect()

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# OBS: se tiver removido df_base_classes lá em cima, remova também este merge
# ou adapte para carregar flag_especializado de outra fonte. Caso mantenha,
# não dê del em df_base_classes antes deste ponto.

df_teste_com_classe_pred = PreverClasseTeste_Modelo7(
    df_teste_all,
    list_features_modelo_7_2,
    modelo_7_2_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não será mais usado
del df_teste_all
gc.collect()

df_validacao_7_2 = RealizarPredicaoTeste_Modelo7(
    df_teste_com_classe_pred,
    list_features_modelo_7_2,
    target,
    dict_modelo_7_2_regressao_geral,
    dict_modelo_7_2_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred
gc.collect()

# ===== Métricas =====
metricas_modelo_7_2 = CalcularMetricasTeste_Modelo7(
    df_validacao_7_2,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

# ===== Salvando =====
SalvarValidacaoModeloMetricas_Modelo7(
    df_validacao_7_2,
    modelo_7_2_classificacao,
    dict_modelo_7_2_regressao_geral,
    dict_modelo_7_2_regressao_especializado,
    metricas_modelo_7_2,
    target,
    modelo_number,
    final_db=final_db
)

# Após salvar, podemos liberar praticamente tudo
del (
    df_validacao_7_2,
    modelo_7_2_classificacao,
    dict_modelo_7_2_regressao_geral,
    dict_modelo_7_2_regressao_especializado,
    metricas_modelo_7_2
)
gc.collect()

# %% [markdown]
# ## Modelo 7.3

# %%
# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,23), # 23 é o top 5% da base de dados
    2:(23,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}

list_features_modelo_7_3 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

modelo_number = '7_3'

# ===== Carregando base =====
df_base = ImportBase_Modelo7()

df_base_classes = CriarColunaClasse_Modelo7(df_base, dict_classes)

df_base_classes['flag_especializado'] = (df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo7(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo7(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# Já podemos descartar as bases completas
del df_base, df_base_classes, df_base_geral, df_base_especializado
gc.collect()

# ===== Treino do modelo de classificação (usando df_treino_geral com todas as classes) =====
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_7_3,
    target_classificacao
)

modelo_7_3_classificacao = TreinarClassificacao_Modelo7(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Libera X e y de classificação (não serão mais usados)
del df_X_treino_classificacao, df_y_treino_classificacao
gc.collect()

# ===== Treino dos modelos de regressão por classe - geral e especializado =====
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_3, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_3, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

dict_modelo_7_3_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_7_3_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# Podemos liberar tudo que era de treino
del (
    df_treino_geral,
    df_treino_especializado,
    dict_treino_regressao_geral,
    dict_treino_regressao_especializado,
    dict_X_y_treino_regressao_geral,
    dict_X_y_treino_regressao_especializado
)
gc.collect()

# ===== Validação =====
df_teste_all = pd.concat([df_teste_geral, df_teste_especializado], ignore_index=True).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais necessários depois de df_teste_all
del df_teste_geral, df_teste_especializado
gc.collect()

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# OBS: se tiver removido df_base_classes lá em cima, remova também este merge
# ou adapte para carregar flag_especializado de outra fonte. Caso mantenha,
# não dê del em df_base_classes antes deste ponto.

df_teste_com_classe_pred = PreverClasseTeste_Modelo7(
    df_teste_all,
    list_features_modelo_7_3,
    modelo_7_3_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não será mais usado
del df_teste_all
gc.collect()

df_validacao_7_3 = RealizarPredicaoTeste_Modelo7(
    df_teste_com_classe_pred,
    list_features_modelo_7_3,
    target,
    dict_modelo_7_3_regressao_geral,
    dict_modelo_7_3_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred
gc.collect()

# ===== Métricas =====
metricas_modelo_7_3 = CalcularMetricasTeste_Modelo7(
    df_validacao_7_3,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

# ===== Salvando =====
SalvarValidacaoModeloMetricas_Modelo7(
    df_validacao_7_3,
    modelo_7_3_classificacao,
    dict_modelo_7_3_regressao_geral,
    dict_modelo_7_3_regressao_especializado,
    metricas_modelo_7_3,
    target,
    modelo_number,
    final_db=final_db
)

# Após salvar, podemos liberar praticamente tudo
del (
    df_validacao_7_3,
    modelo_7_3_classificacao,
    dict_modelo_7_3_regressao_geral,
    dict_modelo_7_3_regressao_especializado,
    metricas_modelo_7_3
)
gc.collect()

# %% [markdown]
# ## Modelo 7.4

# %%
# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,51), # 51 é o top 1% da base de dados
    3:(51,np.inf)
}

list_features_modelo_7_4 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

modelo_number = '7_4'

# ===== Carregando base =====
df_base = ImportBase_Modelo7()

df_base_classes = CriarColunaClasse_Modelo7(df_base, dict_classes)

df_base_classes['flag_especializado'] = (df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo7(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo7(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# Já podemos descartar as bases completas
del df_base, df_base_classes, df_base_geral, df_base_especializado
gc.collect()

# ===== Treino do modelo de classificação (usando df_treino_geral com todas as classes) =====
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_7_4,
    target_classificacao
)

modelo_7_4_classificacao = TreinarClassificacao_Modelo7(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Libera X e y de classificação (não serão mais usados)
del df_X_treino_classificacao, df_y_treino_classificacao
gc.collect()

# ===== Treino dos modelos de regressão por classe - geral e especializado =====
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_4, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_4, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

dict_modelo_7_4_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_7_4_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# Podemos liberar tudo que era de treino
del (
    df_treino_geral,
    df_treino_especializado,
    dict_treino_regressao_geral,
    dict_treino_regressao_especializado,
    dict_X_y_treino_regressao_geral,
    dict_X_y_treino_regressao_especializado
)
gc.collect()

# ===== Validação =====
df_teste_all = pd.concat([df_teste_geral, df_teste_especializado], ignore_index=True).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais necessários depois de df_teste_all
del df_teste_geral, df_teste_especializado
gc.collect()

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# OBS: se tiver removido df_base_classes lá em cima, remova também este merge
# ou adapte para carregar flag_especializado de outra fonte. Caso mantenha,
# não dê del em df_base_classes antes deste ponto.

df_teste_com_classe_pred = PreverClasseTeste_Modelo7(
    df_teste_all,
    list_features_modelo_7_4,
    modelo_7_4_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não será mais usado
del df_teste_all
gc.collect()

df_validacao_7_4 = RealizarPredicaoTeste_Modelo7(
    df_teste_com_classe_pred,
    list_features_modelo_7_4,
    target,
    dict_modelo_7_4_regressao_geral,
    dict_modelo_7_4_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred
gc.collect()

# ===== Métricas =====
metricas_modelo_7_4 = CalcularMetricasTeste_Modelo7(
    df_validacao_7_4,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

# ===== Salvando =====
SalvarValidacaoModeloMetricas_Modelo7(
    df_validacao_7_4,
    modelo_7_4_classificacao,
    dict_modelo_7_4_regressao_geral,
    dict_modelo_7_4_regressao_especializado,
    metricas_modelo_7_4,
    target,
    modelo_number,
    final_db=final_db
)

# Após salvar, podemos liberar praticamente tudo
del (
    df_validacao_7_4,
    modelo_7_4_classificacao,
    dict_modelo_7_4_regressao_geral,
    dict_modelo_7_4_regressao_especializado,
    metricas_modelo_7_4
)
gc.collect()

# %% [markdown]
# ## Modelo 7.5

# %%
# Parâmetros de treinamento
pct_train_test_split = 0.7
threshold_modelo_especializado = 0.5
target = 'vl_precipitacao'
target_classificacao = 'classe_precipitacao'
coluna_vl_prioridade_vizinha = 'vl_prioridade_vizinha_1'

dict_classes = {
    0:(0,1),
    1:(1,7), # 7 é o top 15% da base de dados
    2:(7,np.inf)
}

list_features_modelo_7_5 = list_features_geoespaciais + list_features_estacoes + list_features_vizinhas + list_features_produtos
Algoritmo = XGBRegressor
AlgoritmoClassificacao = XGBClassifier

# Parâmetros de métricas
psc_a_max_chuva = 1
pcc_a_erro = 1
pmc_a_erro = 5
pmc_a_min_chuva = 20

modelo_number = '7_5'

# ===== Carregando base =====
df_base = ImportBase_Modelo7()

df_base_classes = CriarColunaClasse_Modelo7(df_base, dict_classes)

df_base_classes['flag_especializado'] = (df_base_classes[coluna_vl_prioridade_vizinha] >= threshold_modelo_especializado).astype(int)

df_base_geral, df_base_especializado = SepararBaseEspecializadoGeral_Modelo7(
    df_base_classes,
    coluna_vl_prioridade_vizinha,
    threshold_modelo_especializado
)

df_treino_especializado, df_teste_especializado, df_treino_geral, df_teste_geral = SplitTreinoTeste_Modelo7(
    df_base_geral,
    df_base_especializado,
    pct_train_test_split,
    coluna_percentil_temporal='percentil_temporal'
)

# Já podemos descartar as bases completas
del df_base, df_base_classes, df_base_geral, df_base_especializado
gc.collect()

# ===== Treino do modelo de classificação (usando df_treino_geral com todas as classes) =====
df_X_treino_classificacao, df_y_treino_classificacao = PrepararBaseTreino(
    df_treino_geral,
    list_features_modelo_7_5,
    target_classificacao
)

modelo_7_5_classificacao = TreinarClassificacao_Modelo7(
    df_X_treino_classificacao,
    df_y_treino_classificacao,
    AlgoritmoClassificacao
)

# Libera X e y de classificação (não serão mais usados)
del df_X_treino_classificacao, df_y_treino_classificacao
gc.collect()

# ===== Treino dos modelos de regressão por classe - geral e especializado =====
dict_treino_regressao_geral = {
    classe: df_treino_geral.loc[df_treino_geral[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_treino_regressao_especializado = {
    classe: df_treino_especializado.loc[df_treino_especializado[target_classificacao] == classe]
    for classe in dict_classes.keys()
}

dict_X_y_treino_regressao_geral = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_5, target)
    for classe, df_treino_classe in dict_treino_regressao_geral.items()
    if len(df_treino_classe) > 0
}

dict_X_y_treino_regressao_especializado = {
    classe: PrepararBaseTreino(df_treino_classe, list_features_modelo_7_5, target)
    for classe, df_treino_classe in dict_treino_regressao_especializado.items()
    if len(df_treino_classe) > 0
}

dict_modelo_7_5_regressao_geral = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_geral.items()
}

dict_modelo_7_5_regressao_especializado = {
    classe: TreinarAlgoritmo_Modelo7(df_X_treino, df_y_treino, Algoritmo)
    for classe, (df_X_treino, df_y_treino) in dict_X_y_treino_regressao_especializado.items()
}

# Podemos liberar tudo que era de treino
del (
    df_treino_geral,
    df_treino_especializado,
    dict_treino_regressao_geral,
    dict_treino_regressao_especializado,
    dict_X_y_treino_regressao_geral,
    dict_X_y_treino_regressao_especializado
)
gc.collect()

# ===== Validação =====
df_teste_all = pd.concat([df_teste_geral, df_teste_especializado], ignore_index=True).drop_duplicates(
    subset=['id_estacao', 'dt_medicao']
)

# df_teste_geral / df_teste_especializado não são mais necessários depois de df_teste_all
del df_teste_geral, df_teste_especializado
gc.collect()

if 'flag_especializado' not in df_teste_all.columns:
    df_teste_all = df_teste_all.merge(
        df_base_classes[['id_estacao', 'dt_medicao', 'flag_especializado']],
        on=['id_estacao', 'dt_medicao'],
        how='left'
    )

# OBS: se tiver removido df_base_classes lá em cima, remova também este merge
# ou adapte para carregar flag_especializado de outra fonte. Caso mantenha,
# não dê del em df_base_classes antes deste ponto.

df_teste_com_classe_pred = PreverClasseTeste_Modelo7(
    df_teste_all,
    list_features_modelo_7_5,
    modelo_7_5_classificacao,
    nome_coluna_classe_predita='classe_predita'
)

# df_teste_all não será mais usado
del df_teste_all
gc.collect()

df_validacao_7_5 = RealizarPredicaoTeste_Modelo7(
    df_teste_com_classe_pred,
    list_features_modelo_7_5,
    target,
    dict_modelo_7_5_regressao_geral,
    dict_modelo_7_5_regressao_especializado,
    modelo_number=modelo_number,
    coluna_classe_predita='classe_predita',
    coluna_flag_especializado='flag_especializado'
)

# df_teste_com_classe_pred não é mais necessário
del df_teste_com_classe_pred
gc.collect()

# ===== Métricas =====
metricas_modelo_7_5 = CalcularMetricasTeste_Modelo7(
    df_validacao_7_5,
    target,
    modelo_number,
    psc_a_max_chuva,
    pcc_a_erro,
    pmc_a_erro,
    pmc_a_min_chuva
)

# ===== Salvando =====
SalvarValidacaoModeloMetricas_Modelo7(
    df_validacao_7_5,
    modelo_7_5_classificacao,
    dict_modelo_7_5_regressao_geral,
    dict_modelo_7_5_regressao_especializado,
    metricas_modelo_7_5,
    target,
    modelo_number,
    final_db=final_db
)

# Após salvar, podemos liberar praticamente tudo
del (
    df_validacao_7_5,
    modelo_7_5_classificacao,
    dict_modelo_7_5_regressao_geral,
    dict_modelo_7_5_regressao_especializado,
    metricas_modelo_7_5
)
gc.collect()

# %% [markdown]
# # Comparações

# %%
list_modelo_numbers = [
    '1_1','1_2','1_3','1_4',
    '2_1','2_2','2_3','2_4',
    '3_1','3_2','3_3','3_4','3_5',
    '4_1','4_2','4_3','4_4','4_5',
    '5_1','5_2','5_3','5_4','5_5',
    '6_1','6_2','6_3','6_4','6_5',
    '7_1','7_2','7_3','7_4','7_5']

metricas_modelos = {}
for modelo_number in list_modelo_numbers:
    with open(f'modelos_finais/metricas_{modelo_number}.json','r') as f:
        metricas_modelos[modelo_number] = json.load(f)

pd.DataFrame(metricas_modelos).T.to_excel('metricas_final.xlsx')


