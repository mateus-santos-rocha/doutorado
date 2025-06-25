from pulp import *
from tqdm.notebook import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from itertools import product
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import duckdb
import numpy as np

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
            if i in final_selected_indices[id_estacao_base]
        ]
        for id_estacao_base in estacoes
    }
    return problem_status, final_selected_indices, final_deviations, final_objective, id_estacoes_vizinhas


def executar_teste(i, params, estacoes,distancias_dict,intersecoes_dict,prefix):
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

    caminho_arquivo = os.path.join("testes_goal_programming", f"{prefix}_{i}.json")
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
    intersecoes_dict,
    prefix
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
        resultado = executar_teste(i + 1, params, estacoes,distancias_dict,intersecoes_dict,prefix)
        print(resultado)

def gerar_relatorio_vizinhas(
    id_estacoes_vizinhas: dict[str, list[str | int]],
    df_distancias: pd.DataFrame,
    limite_minimo: int = 3
):
    """
    Gera um panorama geral das estações vizinhas selecionadas para cada estação base.

    Args:
        id_estacoes_vizinhas (dict): dicionário com chave = id_estacao_base, valor = lista de vizinhas.
        df_distancias (pd.DataFrame): dataframe com colunas id_estacao_base, id_estacao_candidata, vl_distancia_km.
        limite_minimo (int): valor mínimo de vizinhas para destacar estações com poucas vizinhas.

    Returns:
        None (apenas imprime o relatório)
    """
    num_bases = len(id_estacoes_vizinhas)
    total_vizinhas = sum(len(v) for v in id_estacoes_vizinhas.values())
    media_vizinhas = total_vizinhas / num_bases if num_bases > 0 else 0
    max_vizinhas = max(len(v) for v in id_estacoes_vizinhas.values())
    min_vizinhas = min(len(v) for v in id_estacoes_vizinhas.values())

    estacoes_com_poucas_vizinhas = [
        est for est, viz in id_estacoes_vizinhas.items() if len(viz) < limite_minimo
    ]

    todas_vizinhas = [v for viz_list in id_estacoes_vizinhas.values() for v in viz_list]
    frequencia_vizinhas = Counter(todas_vizinhas)

    # 🔥 Parte otimizada: criar DataFrame de pares e fazer merge
    df_vizinhas = pd.DataFrame([
        {"id_estacao_base": base, "id_estacao_candidata": candidata}
        for base, candidatas in id_estacoes_vizinhas.items()
        for candidata in candidatas
    ])

    # Garante que os tipos são compatíveis para o merge
    df_vizinhas["id_estacao_base"] = df_vizinhas["id_estacao_base"].astype(str)
    df_vizinhas["id_estacao_candidata"] = df_vizinhas["id_estacao_candidata"].astype(str)
    df_distancias["id_estacao_base"] = df_distancias["id_estacao_base"].astype(str)
    df_distancias["id_estacao_candidata"] = df_distancias["id_estacao_candidata"].astype(str)

    df_merged = df_vizinhas.merge(
        df_distancias,
        on=["id_estacao_base", "id_estacao_candidata"],
        how="left"
)

    if not df_merged.empty and df_merged["vl_distancia_km"].notna().any():
        distancia_max = df_merged["vl_distancia_km"].max()
        media_por_base = df_merged.groupby("id_estacao_base")["vl_distancia_km"].mean()
        media_das_medias = media_por_base.mean()
    else:
        distancia_max = None
        media_das_medias = None

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
    print("-" * 50)
    print("📏 DISTÂNCIAS ENTRE ESTAÇÕES")
    if distancia_max is not None:
        print(f"Maior distância entre base e vizinha: {distancia_max:.2f} km")
        print(f"Média das médias de distância por estação base: {media_das_medias:.2f} km")
    else:
        print("Não foi possível calcular distâncias (dados ausentes ou incompatíveis).")
    print("=" * 50)


def get_resumo_tentativa(numero_tentativas_total,path_prefix,output_path):

    if os.path.isfile(output_path):
        df_grouped = pd.read_csv(output_path)
    else:
        
        numeros_tentativas = range(1,numero_tentativas_total+1)
        paths = {n_tentativa:f'{path_prefix}{n_tentativa}.json' for n_tentativa in numeros_tentativas}
        tentativas = {}
        for n_tentativa,path in paths.items():
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                tentativas[n_tentativa] = data


        ouro_conn = duckdb.connect('ouro_db')
        df_distancias = ouro_conn.execute("select * from fato_estacoes_distancia").fetchdf()
        df_intersecoes = ouro_conn.execute("select * from fato_estacoes_intersecao").fetchdf()
        ouro_conn.close()
        df_grouped = None

        for n_tentativa,tentativa in tqdm(tentativas.items(),desc='Obtendo informações resumidas das tentativas',leave=False):

            df_vizinhas = pd.DataFrame([
                            {"id_estacao_base": base, "id_estacao_candidata": candidata}
                            for base, candidatas in tentativa['id_estacoes_vizinhas'].items()
                            for candidata in candidatas
                        ])

            df_vizinhas['id_estacao_base'] = df_vizinhas['id_estacao_base'].astype(int)
            df_vizinhas['id_estacao_candidata'] = df_vizinhas['id_estacao_candidata'].astype(int)

            df_merged = df_vizinhas \
                .merge(df_distancias,on=['id_estacao_base','id_estacao_candidata'],how='inner') \
                .merge(df_intersecoes,on=['id_estacao_base','id_estacao_candidata'],how='inner')
            
            df_grouped_current = df_merged.groupby('id_estacao_base',as_index=False).agg(
                n_estacoes_vizinhas = ('id_estacao_candidata','count'),
                mean_vl_distancia_km = ('vl_distancia_km','mean'),
                max_vl_distancia_km = ('vl_distancia_km','max'),
                mean_pct_intersecao_precipitacao = ('pct_intersecao_precipitacao','mean'),
                min_pct_intersecao_precipitacao = ('pct_intersecao_precipitacao','min')
            )
            df_grouped_current['n_tentativa'] = n_tentativa

            for k,v in tentativa['Parametros'].items():
                df_grouped_current[f'param_{k}'] = v if not v is None else np.nan

                

            if df_grouped is None:
                df_grouped = df_grouped_current.copy()
            else:
                df_grouped = pd.concat([df_grouped,df_grouped_current])

        df_grouped.to_csv(output_path,index=False)
            
    return df_grouped






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





