from __future__ import annotations
import argparse, sys, numpy as np, json
from .cm import load_cm_table
from .geometry import read_xyz
from .qgrid import q_grid_f90
from .scattering import intensity_molecular_xray, intensity_j2_xray

def main(argv=None):
    p = argparse.ArgumentParser(description="Gas-phase IAM (X-ray) intensity (F90-compatible), with optional xraydb backend")
    p.add_argument("--xyz", required=True, help="Input XYZ file (labels must match backend keys: affl or xraydb ions)")
    p.add_argument("--nq", type=int, default=1000, help="Number of q points (default 1000)")
    p.add_argument("--qmin", type=float, default=0.0, help="q min [Å^-1] (default 0.0)")
    p.add_argument("--qmax", type=float, default=None, help="q max [Å^-1] (default 8/a0 ≈ 15.12)")
    p.add_argument("--backend", type=str, default="affl", choices=["affl","xraydb"], help="Form-factor backend (default: affl)")
    p.add_argument("--ion-map", type=str, default=None, help="JSON mapping for labels (e.g., '{\"Cval\": \"C\", \"Siv\": \"Si4+\"}')")
    p.add_argument("--affl", type=str, default=None, help="Path to affl.txt when backend=affl. Defaults to bundled copy.")
    p.add_argument("--j2", action="store_true", help="Output the anisotropic j2 elastic component I_j2(q) instead of the isotropic I(q)")
    p.add_argument("--out", type=str, default=None, help="Output .txt file (columns: q  I(q)). Defaults to stdout.")
    args = p.parse_args(argv)

    pos, labels = read_xyz(args.xyz)
    q = q_grid_f90(nq=args.nq, qmin=args.qmin, qmax=args.qmax)

    ion_map = json.loads(args.ion_map) if args.ion_map else None
    cm = load_cm_table(args.affl) if args.backend == "affl" else None
    if args.j2:
        Iq = intensity_j2_xray(pos, labels, q, cm, backend=args.backend, ion_map=ion_map)
    else:
        Iq = intensity_molecular_xray(pos, labels, q, cm, backend=args.backend, ion_map=ion_map)

    label = "I_j2(q)" if args.j2 else "I(q)"
    if args.out:
        np.savetxt(args.out, np.column_stack([q, Iq]), header=f"q[1/Å]    {label} backend={args.backend}")
    else:
        for qi, Ii in zip(q, Iq):
            print(f"{qi:12.6f}  {Ii:16.8f}")

if __name__ == "__main__":
    main()
