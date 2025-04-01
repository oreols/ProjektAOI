import pandas as pd
import matplotlib.pyplot as plt

# Wczytanie danych z pliku txt
csv_path = "models/trained_components/resistor/training_metrics.txt"  # dostosuj ścieżkę, jeśli potrzebujesz
df = pd.read_csv(csv_path)

# Sprawdzenie, czy dane są prawidłowo wczytane
print(df.head())

# Utworzenie wykresu z dwiema osiami Y
fig, ax1 = plt.subplots(figsize=(10, 6))

# Wykres dla Avg Loss na lewej osi Y
color = 'tab:blue'
ax1.set_xlabel('Epoka')
ax1.set_ylabel('Avg Loss', color=color)
ax1.plot(df["epoch"], df["avg_loss"], marker="o", color=color, label="Avg Loss")
ax1.tick_params(axis='y', labelcolor=color)

# Utworzenie drugiej osi Y (dzielonej z ax1) dla mAP
ax2 = ax1.twinx()
color = 'tab:green'
ax2.set_ylabel('mAP', color=color)
ax2.plot(df["epoch"], df["mAP"], marker="o", color=color, label="mAP")
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title("Proces uczenia: Avg Loss i mAP")
plt.show()
