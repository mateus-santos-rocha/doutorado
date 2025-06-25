def P_1(correlacao,pct_intersecao,distancia_km,D_medio = 10,fator_decaimento=2):
    base_function = correlacao*pct_intersecao/(1+(distancia_km/D_medio)**fator_decaimento)
    return max(0,base_function)