import duckdb
from stations_selections_utils import testar_multiplos_parametros_gp,precalcular_distancias_intersecoes,get_distancias,get_intersecoes
import os

# Explorando diversos cenários para as estações vizinhas
ouro_conn = duckdb.connect('ouro_db')
os.makedirs("testes_goal_programming", exist_ok=True)

valores_peso_distancia = [1, 2, 5, 10]
valores_peso_intersecao = [1]
valores_min_estacoes_candidatas = [2]
valores_max_estacoes_candidatas = [4]
valores_max_distancia = [20, 50, 100]
valores_min_intersecao = [80, 160]

estacoes = ouro_conn.execute('select distinct id_estacao from dim_estacoes').fetch_df()['id_estacao'].tolist()
distancias_dict, intersecoes_dict = precalcular_distancias_intersecoes(estacoes,get_distancias,get_intersecoes,ouro_conn)

testar_multiplos_parametros_gp(valores_peso_distancia,valores_peso_intersecao,valores_min_estacoes_candidatas,valores_max_estacoes_candidatas,valores_max_distancia,valores_min_intersecao,estacoes,distancias_dict,intersecoes_dict)




