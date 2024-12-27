import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configurações de estilo
sns.set_style("whitegrid")
plt.rcParams.update({
    "axes.facecolor": "#f8f9fa",  # Fundo leve
    "axes.edgecolor": "#dddddd",
    "axes.labelcolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#333333",
    "font.size": 12,
    "figure.figsize": (12, 7)
})


def criar_grafico_avaliacao():
    caminho_arquivo = r'C:\Users\RAYO\PycharmProjects\YOLOV8\yolov8n\yolov8-main\AVALIAÇÃO PRÉ PROCESSAMENTO.csv'
    df = pd.read_csv(caminho_arquivo, encoding='ISO-8859-1')

    plt.figure()
    cores = sns.color_palette("pastel")  # Paleta de cores suaves
    bars = plt.bar(df['pre-processamento'], df['acuracia media'], color=cores)
    plt.xlabel('Pré-processamento', labelpad=10, fontsize=14, fontweight='bold')
    plt.ylabel('Acurácia', labelpad=10, fontsize=14, fontweight='bold')
    plt.title('Avaliação Geral', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Adiciona os valores acima das barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=10)

    plt.show()


def criar_grafico_performance():
    caminho_arquivo = r'C:\Users\RAYO\PycharmProjects\YOLOV8\yolov8n\yolov8-main\performance_data.csv'
    df = pd.read_csv(caminho_arquivo, encoding='ISO-8859-1')

    plt.figure()
    cores = sns.color_palette("deep")  # Paleta de cores mais intensa para diferenciação
    bars = plt.bar(df['pre-processamento'], df['acuracia media'], color=cores)
    plt.xlabel('Pré-processamento', labelpad=10, fontsize=14, fontweight='bold')
    plt.ylabel('Acurácia Média', labelpad=10, fontsize=14, fontweight='bold')
    plt.title('Gráfico Dinâmico', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()

    # Adiciona os valores acima das barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.2f}", ha="center", va="bottom", fontsize=10)

    plt.show()