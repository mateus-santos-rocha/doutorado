import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from metrics import precision,recall,F_score,R2_determinacao,RMSE,MAE,PSC_A,PCC_A,PMC_A


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

        # Adiciona rótulos com caixinha em cima de cada barra
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

    # Remove eixos extras se não forem usados
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
    comparison = X_test[['id_estacao','latitude','longitude','dt_medicao']].copy()
    comparison.loc[:,'y_test'] = y_test.copy()
    comparison.loc[:,'y_pred'] = y_pred.copy()
    return comparison