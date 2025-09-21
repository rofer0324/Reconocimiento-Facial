import numpy as np

# Carga de los archivos
embeddings = np.load("../GUI/embeddings/embeddings.npy")  # Forma: (N, D) -> N samples, D dimensiones
labels = np.load("../GUI/embeddings/labels.npy")          # Forma: (N,) o lista de etiquetas

# Guardar embeddings en TSV
with open("../GUI/embeddings/embeddings.tsv", "w") as f_emb:
    for emb in embeddings:
        f_emb.write("\t".join(map(str, emb)) + "\n")

# Guardar labels en TSV
with open("../GUI/embeddings/labels.tsv", "w") as f_lab:
    for label in labels:
        f_lab.write(str(label) + "\n")

print("✅ Archivos 'embeddings.tsv' y 'labels.tsv' generados correctamente.")
