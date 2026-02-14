from pathlib import Path

def listar_clases_keypoints(keypoints_dir: str):
    keypoints_path = Path(keypoints_dir)

    clases = set()
    for p in keypoints_path.rglob("*.h5"):
        nombre = p.name  # ej: "hola_2.h5"
        if "_" not in nombre:
            continue
        clase = nombre.split("_", 1)[0]
        if clase:
            clases.add(clase)

    return sorted(clases)

if __name__ == "__main__":
    clases = listar_clases_keypoints(r"data\keypoints")
    print("Clases encontradas:")
    for c in clases:
        print(c)