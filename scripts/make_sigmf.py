#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path
from sigmf import SigMFFile

CANDIDATE_EXTS = [".sigmf-data", ".iq", ".cf32", ".c32", ".bin"]


def find_or_create_sigmf_data(stem: Path) -> Path:
    """
    Busca un archivo de datos existente para 'stem' con extensiones conocidas.
    Si encuentra uno que NO sea '.sigmf-data', crea 'stem.sigmf-data' por hardlink o copia.
    Devuelve la ruta final 'stem.sigmf-data'.
    """
    # 1) ¿ya existe el .sigmf-data?
    data_path = stem.with_suffix(".sigmf-data")
    if data_path.exists():
        return data_path

    # 2) Buscar candidatos (orden de preferencia)
    for ext in CANDIDATE_EXTS[1:]:
        p = stem.with_suffix(ext)
        if p.exists():
            # crear/llenar .sigmf-data
            try:
                os.link(p, data_path)  # hardlink si se puede (rápido, sin duplicar)
            except Exception:
                shutil.copyfile(p, data_path)  # sino, copia
            return data_path

    raise FileNotFoundError(
        f"No se encontró archivo de datos para el stem '{stem}'. "
        f"Crea por ejemplo '{stem}.iq' o '{stem}.sigmf-data'."
    )


def load_annotations(path: str | None):
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as fh:
        anns = json.load(fh)
    if not isinstance(anns, list):
        raise ValueError("El JSON de anotaciones debe ser una lista de anotaciones.")
    return anns


def write_sigmf(stem: Path, fs: float, f_center: float, annotations=None, description="ESM wideband + CFAR detections"):
    annotations = annotations or []
    data_path = find_or_create_sigmf_data(stem)
    meta_path = stem.with_suffix(".sigmf-meta")

    f = SigMFFile(
        data_file=str(data_path.resolve()),  # nombre relativo
        global_info={
            "core:datatype": "cf32_le",
            "core:sample_rate": float(fs),
            "core:version": "1.0.0",
            "core:description": description,
        },
    )
    f.add_capture(0, metadata={"core:frequency": float(f_center)})

    # Cada anotación: {sample_start, sample_count, freq_lower, freq_upper, label?}
    for ann in annotations:
        f.add_annotation(
            int(ann["sample_start"]),
            length=int(ann["sample_count"]),
            metadata={
                "core:freq_lower_edge": float(ann["freq_lower"]),
                "core:freq_upper_edge": float(ann["freq_upper"]),
                "core:description": str(ann.get("label", "CFAR detection")),
            },
        )

    f.tofile(meta_path)
    return data_path, meta_path


def main():
    ap = argparse.ArgumentParser(description="Crear par SigMF (.sigmf-data + .sigmf-meta) desde un stem")
    ap.add_argument("iq_stem", help="Ruta sin extensión. Se aceptan <stem>.iq o <stem>.sigmf-data, etc.")
    ap.add_argument("--fs", type=float, required=True, help="Sample rate (Hz)")
    ap.add_argument("--fc", type=float, required=True, help="Frecuencia central (Hz)")
    ap.add_argument("--ann", type=str, default=None, help="JSON con lista de anotaciones (opcional)")
    ap.add_argument("--desc", type=str, default="ESM wideband + CFAR detections", help="Descripción global (opcional)")
    args = ap.parse_args()

    stem = Path(args.iq_stem)
    anns = load_annotations(args.ann)
    data_path, meta_path = write_sigmf(stem, fs=args.fs, f_center=args.fc, annotations=anns, description=args.desc)
    print("OK → data:", data_path)
    print("OK → meta:", meta_path)


if __name__ == "__main__":
    main()
