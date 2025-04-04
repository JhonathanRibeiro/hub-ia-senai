# Aula 12 - 25/03/2025
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import time
from datetime import datetime

# Função para formatar tempo
def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:.0f} min {seconds:.2f} s"

# Leitura dos dados
print("\nCarregando dados...")
df = pd.read_csv("dataset.csv")

# Separando as features (X) e o rótulo (y)
X = df.iloc[:, 1:-1]  
y = df.iloc[:, -1]  

# Convertendo os rótulos para valores numéricos
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalizando os dados
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=62)

# Dicionário de modelos
modelos = {
    "KNN": KNeighborsClassifier(),
    # "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    # "MLP": MLPClassifier(random_state=42, max_iter=1000),
    # "SVM": SVC(random_state=42),
    # "Random Forest": RandomForestClassifier(random_state=42),
    # "AdaBoost": AdaBoostClassifier(random_state=42),
    # "Naive Bayes": GaussianNB()
}

# Tabela de resultados
resultados = []
print("\nIniciando avaliação dos modelos...")

for nome, modelo in modelos.items():
    print(f"\n=== Modelo: {nome} ===")
    # Treinamento
    inicio_treino = time.time()
    print(f"Início treino: {datetime.now().strftime('%H:%M:%S')}")
    modelo.fit(X_train, y_train)
    fim_treino = time.time()
    tempo_treino = fim_treino - inicio_treino
    print(f"Fim treino: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Tempo treino: {format_time(tempo_treino)}")
    # Teste
    inicio_teste = time.time()
    print(f"Início teste: {datetime.now().strftime('%H:%M:%S')}")
    y_pred = modelo.predict(X_test)
    fim_teste = time.time()
    tempo_teste = fim_teste - inicio_teste
    print(f"Fim teste: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Tempo teste: {format_time(tempo_teste)}")
    # Cálculo de métricas
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    resultados.append({
        "Modelo": nome,
        "Acurácia": round(acc, 4),
        "F1-Score": round(f1, 4),
        "Revocação": round(recall, 4),
        "Precisão": round(precision, 4),
        "Tempo Treino (s)": round(tempo_treino, 2),
        "Tempo Teste (s)": round(tempo_teste, 2)
    })

# Criar tabela comparativa
print("\n\n=== RESULTADOS FINAIS ===")
df_resultados = pd.DataFrame(resultados)
print(df_resultados.to_string(index=False))
# Exportar para CSV
df_resultados.to_csv('resultados_modelos_com_tempo.csv', index=False)
print("\nResultados salvos em 'resultados_modelos_com_tempo.csv'")
# Visualização
df_resultados.set_index("Modelo")[["Acurácia", "F1-Score", "Revocação", "Precisão"]].plot(kind='bar', figsize=(12,6))
plt.title("Comparação de Desempenho entre Modelos")
plt.ylabel("Pontuação")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Visualização dos tempos
df_resultados.set_index("Modelo")[["Tempo Treino (s)", "Tempo Teste (s)"]].plot(kind='bar', figsize=(12,6))
plt.title("Tempo de Execução por Modelo")
plt.ylabel("Segundos")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()