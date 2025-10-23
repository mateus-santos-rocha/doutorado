import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats

def plotar_estacoes_mapa(df, markersize=1, markercolor='blue', 
                         xlim=(-54, -37), ylim=(-25, -16), 
                         figsize=(10, 10), mostrar_borda=True):
    """
    Plota um mapa com estações meteorológicas e divisas estaduais.

    Parâmetros:
    - df: DataFrame com colunas 'latitude' e 'longitude'
    - markersize: tamanho dos pontos das estações
    - markercolor: cor dos pontos das estações
    - xlim: tupla com limites de longitude (min, max)
    - ylim: tupla com limites de latitude (min, max)
    - figsize: tupla com tamanho da figura (largura, altura)
    - mostrar_borda: se True, desenha uma borda pontilhada na área (-53, -24) até (-38, -17)
    """

    gdf_estacoes = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    )

    brasil = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    estados = gpd.read_file("https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson")

    brasil = brasil[brasil['name'] == 'Brazil']
    estados = estados.to_crs("EPSG:4326")

    fig, ax = plt.subplots(figsize=figsize)
    brasil.plot(ax=ax, color='white', edgecolor='black')
    estados.boundary.plot(ax=ax, color='gray', linewidth=1)
    gdf_estacoes.plot(ax=ax, color=markercolor, markersize=markersize, label='Estações')

    if mostrar_borda:
        rect = patches.Rectangle(
            xy=(-53, -24), width=15, height=7,  # (lon_min, lat_min), largura, altura
            linewidth=1,
            edgecolor='black',
            facecolor='none',
            linestyle='dotted'
        )
        ax.add_patch(rect)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()
    plt.title('Estações Meteorológicas na Região R2')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()


def plot_distance_correlation(df, title='Relação entre Distância e Correlação', s=50, alpha=0.6, figsize=(10, 6)):
    """
    Plota um scatter plot de vl_distancia_km vs correlacao com linha de best fit.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo as colunas 'vl_distancia_km' e 'correlacao'
    s : float, default=50
        Tamanho dos pontos no scatter plot
    alpha : float, default=0.6
        Transparência dos pontos (0=transparente, 1=opaco)
    figsize : tuple, default=(10, 6)
        Tamanho da figura (largura, altura)
    
    Returns
    -------
    tuple
        (fig, ax): objetos figura e eixo do matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = df['vl_distancia_km'].values
    y = df['correlacao'].values
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    ax.scatter(x_clean, y_clean, s=s, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    x_line = np.array([x_clean.min(), x_clean.max()])
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Best fit')
    
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    ax.set_xlabel('Distância (km)', fontsize=12)
    ax.set_ylabel('Correlação', fontsize=12)
    ax.set_title('Relação entre Distância e Correlação', fontsize=14, fontweight='bold')
    
    textstr = f'Correlação de Pearson: {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, alpha=0.3)
    
    sns.despine()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def plot_distance_correlation_english(df, title='Distance vs Correlation', s=50, alpha=0.6, figsize=(10, 6)):
    """
    Plots a scatter plot of distance vs correlation with best fit line.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'vl_distancia_km' and 'correlacao' columns
    s : float, default=50
        Size of scatter plot points
    alpha : float, default=0.6
        Transparency of points (0=transparent, 1=opaque)
    figsize : tuple, default=(10, 6)
        Figure size (width, height)
    
    Returns
    -------
    tuple
        (fig, ax): matplotlib figure and axes objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = df['vl_distancia_km'].values
    y = df['correlacao'].values
    
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    ax.scatter(x_clean, y_clean, s=s, alpha=alpha, edgecolors='black', linewidth=0.5)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
    
    x_line = np.array([x_clean.min(), x_clean.max()])
    y_line = slope * x_line + intercept
    
    ax.plot(x_line, y_line, 'r-', linewidth=2, label='Best fit')
    
    correlation = np.corrcoef(x_clean, y_clean)[0, 1]
    
    ax.set_xlabel('Distance (km)', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title('Distance vs Correlation', fontsize=14, fontweight='bold')
    
    textstr = f'Pearson Correlation: {correlation:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    ax.grid(True, alpha=0.3)
    
    sns.despine()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax