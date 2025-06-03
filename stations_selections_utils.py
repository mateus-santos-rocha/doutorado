import pandas as pd
import duckdb
from pulp import *
from tqdm.notebook import tqdm
import math
from collections import Counter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import json

def get_distancias(conn, id_estacao_base):
    rows = conn.execute(
        'SELECT vl_distancia_km FROM fato_estacoes_distancia WHERE id_estacao_base = ? ORDER BY id_estacao_candidata',
        [id_estacao_base]
    ).fetchall()
    return [r[0] for r in rows]

def get_intersecoes(conn, id_estacao_base):
    rows = conn.execute(
        'SELECT pct_intersecao_precipitacao FROM fato_estacoes_intersecao WHERE id_estacao_base = ? ORDER BY id_estacao_candidata',
        [id_estacao_base]
    ).fetchall()
    return [r[0] for r in rows]

def precalcular_distancias_intersecoes(estacoes, get_distancias, get_intersecoes, conn):
    distancias_dict = {}
    intersecoes_dict = {}

    for id_estacao_base in tqdm(estacoes, desc="Pré-calculando distâncias e interseções"):
        distancias_dict[id_estacao_base] = get_distancias(conn,id_estacao_base)
        intersecoes_dict[id_estacao_base] = get_intersecoes(conn,id_estacao_base)

    return distancias_dict, intersecoes_dict

from pulp import *

def solve_goal_programming_problem_with_lists(
    num_items: int,
    min_estacoes_candidatas: int | None,
    max_estacoes_candidatas: int | None,
    max_distancia_aceita: float | None,
    peso_distancia: float,
    distancias: list[float],
    min_intersecao_aceito: float | None,
    peso_intersecao: float,
    intersecoes: list[float]
):
    prob = LpProblem("Goal_Programming_Station_Selection", LpMinimize)

    x = [LpVariable(f"x_{j}", cat=LpBinary) for j in range(num_items)]

    d_plus = LpVariable("d_plus", 0)
    e_minus = LpVariable("e_minus", 0)

    weight_d_plus = peso_distancia / max_distancia_aceita if max_distancia_aceita not in (None, 0) else 0
    weight_e_minus = peso_intersecao / min_intersecao_aceito if min_intersecao_aceito not in (None, 0) else 0

    prob += weight_d_plus * d_plus + weight_e_minus * e_minus

    if max_estacoes_candidatas is not None:
        prob += lpSum(x) <= max_estacoes_candidatas
    if min_estacoes_candidatas is not None:
        prob += lpSum(x) >= min_estacoes_candidatas

    if max_distancia_aceita is not None:
        prob += lpSum(distancias[j] * x[j] for j in range(num_items)) <= max_distancia_aceita + d_plus
    else:
        prob += d_plus == 0

    if min_intersecao_aceito is not None:
        prob += lpSum(intersecoes[j] * x[j] for j in range(num_items)) >= min_intersecao_aceito - e_minus
    else:
        prob += e_minus == 0

    prob.solve(PULP_CBC_CMD(msg=0))

    status = LpStatus[prob.status]
    selected = [j for j in range(num_items) if x[j].varValue == 1] if status in ["Optimal", "Feasible"] else []
    deviations = {
        "d_plus": d_plus.varValue,
        "e_minus": e_minus.varValue
    } if status in ["Optimal", "Feasible"] else {}
    objective_value = value(prob.objective) if status in ["Optimal", "Feasible"] else float('inf')

    return status, selected, deviations, objective_value

def selecionar_estacoes_candidatas(
    estacoes,
    distancias_dict,
    intersecoes_dict,
    solve_goal_programming_problem_with_lists,
    peso_distancia,
    peso_intersecao,
    min_estacoes_candidatas,
    max_estacoes_candidatas,
    max_distancia_aceita,
    min_intersecao_aceito,
    verbose=False
):
    """
    Executa o algoritmo de seleção de estações candidatas baseado em programação por metas.

    Args:
        estacoes (list): Lista de IDs das estações.
        solve_goal_programming_problem_with_lists (function): Função de resolução do problema.
        peso_distancia (float): Peso dado à distância no modelo.
        peso_intersecao (float): Peso dado à interseção no modelo.
        min_estacoes_candidatas (int | None): Mínimo de estações candidatas (pode ser None).
        max_estacoes_candidatas (int | None): Máximo de estações candidatas (pode ser None).
        max_distancia_aceita (float): Distância máxima permitida.
        min_intersecao_aceito (float): Valor mínimo de interseção aceitável.
        verbose (bool): Se True, imprime o ID da estação base durante o processo.

    Returns:
        tuple: (problem_status, final_selected_indices, final_deviations, final_objective)
    """
    problem_status = {}
    final_selected_indices = {}
    final_deviations = {}
    final_objective = {}

    for id_estacao_base in tqdm(estacoes, desc="Calculando estações candidatas", leave=False):
        distancias = distancias_dict[id_estacao_base]
        intersecoes = intersecoes_dict[id_estacao_base]
        id_estacoes_candidatas_list = [eid for eid in estacoes if eid != id_estacao_base]
        id_estacoes_candidatas_list.sort()
        distancias = distancias
        intersecoes = intersecoes
        num_itens = len(id_estacoes_candidatas_list)

        if verbose:
            print(f'id_estacao_base: {id_estacao_base}')

        resultado = solve_goal_programming_problem_with_lists(
            num_items=num_itens,
            min_estacoes_candidatas=min_estacoes_candidatas,
            max_estacoes_candidatas=max_estacoes_candidatas,
            max_distancia_aceita=max_distancia_aceita,
            peso_distancia=peso_distancia,
            distancias=distancias,
            min_intersecao_aceito=min_intersecao_aceito,
            peso_intersecao=peso_intersecao,
            intersecoes=intersecoes
        )

        problem_status[id_estacao_base], final_selected_indices[id_estacao_base], final_deviations[id_estacao_base], final_objective[id_estacao_base] = resultado
    
    id_estacoes_vizinhas = {
        id_estacao_base: [
            id_estacao_candidata
            for i, id_estacao_candidata in enumerate(sorted([eid for eid in estacoes if eid != id_estacao_base]))
            if i in final_selected_indices[id_estacao_base].keys()
        ]
        for id_estacao_base in estacoes
    }
    return problem_status, final_selected_indices, final_deviations, final_objective, id_estacoes_vizinhas


def executar_teste(i, params, estacoes,distancias_dict,intersecoes_dict):
    peso_distancia, peso_intersecao, min_estacoes_candidatas, max_estacoes_candidatas, max_distancia_aceita, min_intersecao_aceito = params

    try:
        problem_status, final_selected_indices, final_deviations, final_objective, id_estacoes_vizinhas = selecionar_estacoes_candidatas(
            estacoes=estacoes,
            distancias_dict=distancias_dict,
            intersecoes_dict=intersecoes_dict,
            solve_goal_programming_problem_with_lists=solve_goal_programming_problem_with_lists,
            peso_distancia=peso_distancia,
            peso_intersecao=peso_intersecao,
            min_estacoes_candidatas=min_estacoes_candidatas,
            max_estacoes_candidatas=max_estacoes_candidatas,
            max_distancia_aceita=max_distancia_aceita,
            min_intersecao_aceito=min_intersecao_aceito,
            verbose=False
        )

        resultado = {
            "Tentativa": f"Tentativa {i}",
            "Parametros": {
                "peso_distancia": peso_distancia,
                "peso_intersecao": peso_intersecao,
                "min_estacoes_candidatas": min_estacoes_candidatas,
                "max_estacoes_candidatas": max_estacoes_candidatas,
                "max_distancia_aceita": max_distancia_aceita,
                "min_intersecao_aceito": min_intersecao_aceito
            },
            "problem_status": problem_status,
            "final_selected_indices": final_selected_indices,
            "final_deviations": final_deviations,
            "final_objective": final_objective,
            "id_estacoes_vizinhas": id_estacoes_vizinhas
        }

    except Exception as e:
        resultado = {
            "Tentativa": f"Tentativa {i}",
            "Parametros": {
                "peso_distancia": peso_distancia,
                "peso_intersecao": peso_intersecao,
                "min_estacoes_candidatas": min_estacoes_candidatas,
                "max_estacoes_candidatas": max_estacoes_candidatas,
                "max_distancia_aceita": max_distancia_aceita,
                "min_intersecao_aceito": min_intersecao_aceito
            },
            "erro": str(e)
        }

    caminho_arquivo = os.path.join("testes_goal_programming", f"tentativa_{i}.json")
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)
    with open(caminho_arquivo, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)

    return f"Tentativa {i} finalizada"

def testar_multiplos_parametros_gp(
    valores_peso_distancia,
    valores_peso_intersecao,
    valores_min_estacoes_candidatas,
    valores_max_estacoes_candidatas,
    valores_max_distancia,
    valores_min_intersecao,
    estacoes,
    distancias_dict,
    intersecoes_dict
):
    """
    Testa múltiplas combinações de parâmetros sequencialmente para seleção de estações via programação por metas.
    
    Parâmetros:
    -----------
    (mesmos da versão anterior, exceto max_workers)
    """
    combinacoes = list(product(
        valores_peso_distancia,
        valores_peso_intersecao,
        valores_min_estacoes_candidatas,
        valores_max_estacoes_candidatas,
        valores_max_distancia,
        valores_min_intersecao
    ))

    for i, params in enumerate(tqdm(combinacoes, desc="Testando combinações de parâmetros")):
        resultado = executar_teste(i + 1, params, estacoes,distancias_dict,intersecoes_dict)
        print(resultado)

    


def gerar_relatorio_vizinhas(id_estacoes_vizinhas: dict[str, list[str | int]], limite_minimo=3):
    """
    Gera um panorama geral das estações vizinhas selecionadas para cada estação base.

    Args:
        id_estacoes_vizinhas (dict): dicionário com chave = id_estacao_base, valor = lista de vizinhas.
        limite_minimo (int): valor mínimo de vizinhas para destacar estações com poucas vizinhas.

    Returns:
        None (apenas imprime o relatório)
    """
    num_bases = len(id_estacoes_vizinhas)
    total_vizinhas = sum(len(v) for v in id_estacoes_vizinhas.values())
    media_vizinhas = total_vizinhas / num_bases if num_bases > 0 else 0
    max_vizinhas = max(len(v) for v in id_estacoes_vizinhas.values())
    min_vizinhas = min(len(v) for v in id_estacoes_vizinhas.values())

    # Estações base com menos de X vizinhas
    estacoes_com_poucas_vizinhas = [
        est for est, viz in id_estacoes_vizinhas.items() if len(viz) < limite_minimo
    ]

    # Frequência com que cada estação vizinha foi escolhida
    todas_vizinhas = [v for viz_list in id_estacoes_vizinhas.values() for v in viz_list]
    frequencia_vizinhas = Counter(todas_vizinhas)

    print("📊 RELATÓRIO GERAL DAS ESTAÇÕES VIZINHAS")
    print("=" * 50)
    print(f"Total de estações base analisadas: {num_bases}")
    print(f"Total de estações vizinhas atribuídas: {total_vizinhas}")
    print(f"Média de vizinhas por estação base: {media_vizinhas:.2f}")
    print(f"Máximo de vizinhas atribuídas a uma base: {max_vizinhas}")
    print(f"Mínimo de vizinhas atribuídas a uma base: {min_vizinhas}")
    print("-" * 50)
    print(f"Número de estações base com menos de {limite_minimo} vizinhas: {len(estacoes_com_poucas_vizinhas)}")
    if estacoes_com_poucas_vizinhas:
        print("IDs das estações base com poucas vizinhas:", estacoes_com_poucas_vizinhas)
    print("-" * 50)
    print("Top 10 estações mais frequentemente escolhidas como vizinhas:")
    for estacao, freq in frequencia_vizinhas.most_common(10):
        print(f"  Estação {estacao}: {freq} vezes")
    print("=" * 50)



def plotar_estacoes(estacoes_dict):
    # Limites do mapa
    min_lon, max_lon = -54, -37
    min_lat, max_lat = -25, -16

    # Extrai IDs e coordenadas
    ids = list(estacoes_dict.keys())
    lats = [estacoes_dict[i]['latitude'] for i in ids]
    lons = [estacoes_dict[i]['longitude'] for i in ids]

    # Setup do mapa
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    # Adiciona elementos ao mapa
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.gridlines(draw_labels=True)

    # Plota a primeira estação em vermelho
    ax.plot(lons[0], lats[0], marker='o', color='red', markersize=8, label=f'ID {ids[0]} (principal)')

    # Plota as demais em azul
    if len(ids) > 1:
        ax.scatter(lons[1:], lats[1:], color='blue', label='Demais estações')

    # Adiciona legenda
    ax.legend()
    plt.title('Localização das Estações Meteorológicas')
    plt.show()

def plotar_estacoes_por_id(id_estacao_base, ouro_conn, id_estacoes_vizinhas):
    # 1. Monta lista de IDs
    ids_vizinhos = id_estacoes_vizinhas[id_estacao_base]
    ids_todos = [id_estacao_base] + ids_vizinhos
    ids_str = ','.join(map(str, ids_todos))
    
    # 2. Busca lat/lon do banco
    query = f"""
    SELECT DISTINCT
        id_estacao,
        latitude,
        longitude
    FROM dim_estacoes
    WHERE id_estacao IN ({ids_str})
    """
    df_estacoes = ouro_conn.execute(query).fetch_df().set_index('id_estacao')
    estacoes_dict = df_estacoes.to_dict(orient='index')
    
    # 3. Prepara coordenadas
    ids = list(estacoes_dict.keys())
    lats = [estacoes_dict[i]['latitude'] for i in ids]
    lons = [estacoes_dict[i]['longitude'] for i in ids]
    
    # 4. Plota o mapa
    min_lon, max_lon = -54, -37
    min_lat, max_lat = -25, -16

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.STATES, linestyle=':')
    ax.gridlines(draw_labels=True)

    ax.plot(estacoes_dict[id_estacao_base]['longitude'],
            estacoes_dict[id_estacao_base]['latitude'],
            marker='o', color='red', markersize=8, label=f'Base {id_estacao_base}')

    ids_vizinhos_plot = [i for i in ids if i != id_estacao_base]
    lats_vizinhos = [estacoes_dict[i]['latitude'] for i in ids_vizinhos_plot]
    lons_vizinhos = [estacoes_dict[i]['longitude'] for i in ids_vizinhos_plot]

    if lats_vizinhos:
        ax.scatter(lons_vizinhos, lats_vizinhos, color='blue', label='Estações vizinhas')

    ax.legend(bbox_to_anchor=(1.3,1))
    plt.title('Localização da Estação Base e Vizinhas')
    plt.show()

# estacao 920231 ta com vizinhas mt distantes
# estacao 920840 ta indicando ela mesma como vizinha
# > Talvez duplicidade na base?
# Parametros para possivel escolha 
# > Usar dado de produto como "estação vizinha"? Ver correlação
# >> Correlação forte --> Pode ser usado para o solo
# 
#
#
# BASE: 2025-01-02, *2025-01-01*
# VIZINHA: *2025-01-01*, 2024-12-31
# PCT_INTERSECAO = 50%
#
#
# Para a próxima sexta: trazer cenários de conjuntos de parâmetros
# 





