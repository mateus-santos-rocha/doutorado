import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from metrics import precision,recall,F_score,R2_determinacao,RMSE,MAE,PSC_A,PCC_A,PMC_A
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd


def plot_model_metrics(metrics_df, metric_names):
    """
    Plota gráficos de barras lado a lado comparando métricas entre diferentes modelos.
    Cada barra recebe um rótulo com valor dentro de uma caixa discreta.
    """
    num_metrics = len(metric_names)
    cols = 4
    rows = math.ceil(num_metrics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        if metric not in metrics_df.columns:
            print(f"Métrica '{metric}' não encontrada no DataFrame.")
            continue

        ax = axes[i]
        plot = sns.barplot(
            data=metrics_df,
            x='model_number',
            y=metric,
            hue='model_number',
            palette='crest',
            ax=ax
        )

        ax.set_title(metric)
        ax.set_xlabel('Modelo')
        ax.set_ylabel(metric)
        ax.grid(True, which='major', axis='y', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend_.remove()

        for container in plot.containers:
            for bar in container:
                height = bar.get_height()
                if not math.isnan(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f'{height:.3f}',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='gray',
                            linewidth=0.5,
                            alpha=0.7
                        )
                    )

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def import_model_and_comparison(model_path,comparison_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(comparison_path,'rb') as f:
        comparison = pickle.load(f)
    return model,comparison

def compute_metrics(y_test,y_pred,relative_error_threshold = 0.2,absolute_error_threshold = 10,min_sem_chuva = 1,min_relevancia = 0.5,beta = 1,minimo_muita_chuva = 20):
    metrics = {
        'precision':precision(y_test,y_pred,error_threshold=relative_error_threshold,min_relevancia=min_relevancia),
        'recall':recall(y_test,y_pred,error_threshold=relative_error_threshold,min_relevancia=min_relevancia),
        'F1-Score':F_score(y_test,y_pred,beta=beta,error_threshold=relative_error_threshold,min_relevancia=min_relevancia),
        'RMSE':RMSE(y_test,y_pred),
        'R2':R2_determinacao(y_test,y_pred),
        'MAE':MAE(y_test,y_pred),
        'PSC_A':PSC_A(y_test,y_pred,erro=min_sem_chuva),
        'PCC_A':PCC_A(y_test,y_pred,erro=absolute_error_threshold),
        'PMC_A':PMC_A(y_test,y_pred,erro=absolute_error_threshold,chuva_minima=minimo_muita_chuva)}
    return metrics
    
    
def compute_comparison_df(X_test,y_test,y_pred):
    if 'vl_prioridade_vizinha' not in X_test.columns:
        comparison = X_test[['id_estacao','latitude','longitude','dt_medicao']].copy()
    else:
        comparison = X_test[['id_estacao','latitude','longitude','dt_medicao','vl_prioridade_vizinha']].copy()
    comparison.loc[:,'y_test'] = y_test.copy()
    comparison.loc[:,'y_pred'] = y_pred.copy()
    return comparison

def plot_metrica_heatmap(df_metricas, metrica, model_number, figsize=(10, 8), markersize=40):
    metricas_validas = {'precision', 'recall', 'F1-Score', 'RMSE', 'R2', 'MAE', 'PSC_A', 'PCC_A', 'PMC_A'}
    if metrica not in metricas_validas:
        raise ValueError(f"Métrica inválida. Escolha uma das seguintes: {metricas_validas}")

    metricas_maior_melhor = {'precision', 'recall', 'F1-Score', 'R2', 'PSC_A', 'PCC_A', 'PMC_A'}

    df_model = df_metricas[df_metricas['model_number'] == model_number].copy()

    df_filtrado = df_model[
        (df_model['longitude'] >= -54) & (df_model['longitude'] <= -37) &
        (df_model['latitude'] >= -25) & (df_model['latitude'] <= -16)
    ].copy()

    if metrica == "R2":
        df_filtrado[metrica] = df_filtrado[metrica].clip(lower=0)

    geometry = [Point(xy) for xy in zip(df_filtrado['longitude'], df_filtrado['latitude'])]
    gdf = gpd.GeoDataFrame(df_filtrado, geometry=geometry, crs="EPSG:4326")

    brasil_estados = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    brasil_estados = brasil_estados[brasil_estados['name'] == 'Brazil']
    estados = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")

    cmap = 'RdYlGn' if metrica in metricas_maior_melhor else 'RdYlGn_r'

    fig, ax = plt.subplots(figsize=figsize)
    estados.boundary.plot(ax=ax, linewidth=1, edgecolor='black')
    brasil_estados.plot(ax=ax, color='white', edgecolor='black') 

    gdf.plot(column=metrica, ax=ax, cmap=cmap, legend=True,
             legend_kwds={'label': metrica, 'shrink': 0.6}, markersize=markersize, alpha=1)

    ax.set_xlim(-54, -37)
    ax.set_ylim(-25, -16)
    ax.set_title(f"Distribuição da métrica {metrica} (modelo {model_number})", fontsize=14)

    ax.set_xticks(range(-54, -36, 2))
    ax.set_yticks(range(-25, -15, 2))
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.grid(True, which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_metric_by_bin(
    df_all, 
    metrica, 
    compute_metrics=compute_metrics,  
    figsize=(12, 8)
):
    metricas_validas = {'precision', 'recall', 'F1-Score', 'RMSE', 'R2', 'MAE', 'PSC_A', 'PCC_A', 'PMC_A'}
    if metrica not in metricas_validas:
        raise ValueError(f"Métrica inválida. Escolha uma das seguintes: {metricas_validas}")

    resultados = []
    grouped = df_all.groupby(['model', 'bin_prioridade'])
    for (model, bin_label), group in grouped:
        if len(group) < 2:
            continue
        y_true = group['y_test']
        y_pred = group['y_pred']
        metrics = compute_metrics(y_true, y_pred)
        metrics.update({'model': model, 'bin_prioridade': bin_label})
        resultados.append(metrics)
    df_metricas_bin_model = pd.DataFrame(resultados)

    resultados_overall = []
    grouped_overall = df_all.groupby('model')
    for model, group in grouped_overall:
        if len(group) < 2:
            continue
        y_true = group['y_test']
        y_pred = group['y_pred']
        metrics = compute_metrics(y_true, y_pred)
        resultados_overall.append({'model': model, **metrics})
    df_metricas_overall = pd.DataFrame(resultados_overall)

    plt.figure(figsize=figsize)
    g = sns.barplot(data=df_metricas_bin_model, x='bin_prioridade', y=metrica, hue='model')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    g.set_axisbelow(True)

    palette = sns.color_palette(n_colors=len(df_metricas_bin_model['model'].unique()))
    model_order = sorted(df_metricas_bin_model['model'].unique())

    hue_colors = dict(zip(model_order, palette))

    ax = plt.gca()
    for _, row in df_metricas_overall.iterrows():
        model = row['model']
        val = row[metrica]
        if model in hue_colors:
            ax.axhline(val, color=hue_colors[model], linestyle='--', alpha=0.8)


    plt.title(f'Métrica {metrica} por bin_prioridade e model')
    plt.ylabel(metrica)
    plt.xlabel('Bin prioridade')
    plt.tight_layout()
    plt.show()
