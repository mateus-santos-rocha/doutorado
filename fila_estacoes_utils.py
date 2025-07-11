import numpy as np

def P_1(correlacao, pct_intersecao, distancia_km, D_medio=10, fator_decaimento=2):
    base_function = correlacao * pct_intersecao / (1 + (distancia_km / D_medio) ** fator_decaimento)
    return np.maximum(0, base_function)

def apply_P_as_column(fato_estacoes_base_fila_prioridade,P,D_medio=15,fator_decaimento=2):
    df = fato_estacoes_base_fila_prioridade.copy()
    df["P"] = P(
        correlacao=df["correlacao"].values,
        pct_intersecao=(df["pct_intersecao_precipitacao"]/100).values,
        distancia_km=df["vl_distancia_km"].values,
        D_medio=D_medio,
        fator_decaimento=fator_decaimento
    )
    return df["P"]
    