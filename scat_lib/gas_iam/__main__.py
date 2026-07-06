from __future__ import annotations
import argparse, sys, numpy as np, json
from .cm import load_cm_table
from .geometry import read_xyz
from .qgrid import q_grid_f90
from .scattering import intensity_molecular_xray, intensity_j2_xray

def main(argv=None):
    p = argparse.ArgumentParser(description="Gas-phase IAM (X-ray) intensity, with optional xraydb backend. "
                                            "Form factors are normalised to f(0)=Z by default; pass --no-normalize "
                                            "for the raw tabulated factors (e.g. F90 bit-parity).")
    p.add_argument("--xyz", required=True, help="Input XYZ file (labels must match backend keys: affl or xraydb ions)")
    p.add_argument("--nq", type=int, default=1000, help="Number of q points (default 1000)")
    p.add_argument("--qmin", type=float, default=0.0, help="q min [Å^-1] (default 0.0)")
    p.add_argument("--qmax", type=float, default=None, help="q max [Å^-1] (default 8/a0 ≈ 15.12)")
    p.add_argument("--backend", type=str, default="affl", choices=["affl","xraydb"], help="Form-factor backend (default: affl)")
    p.add_argument("--ion-map", type=str, default=None, help="JSON mapping for labels (e.g., '{\"Cval\": \"C\", \"Siv\": \"Si4+\"}')")
    p.add_argument("--affl", type=str, default=None, help="Path to affl.txt when backend=affl. Defaults to bundled copy.")
    p.add_argument("--j2", action="store_true", help="Output the anisotropic j2 elastic component I_j2(q) instead of the isotropic I(q)")
    p.add_argument("--inelastic", type=str, default=None, choices=["table", "xraydb", "auto"],
                   help="Also compute the inelastic (incoherent) component; adds I_inel(q) and I_tot(q) columns. Not available with --j2.")
    p.add_argument("--no-normalize", action="store_true",
                   help="Use the raw tabulated form factors instead of rescaling them so that f(0) = Z - charge (which enforces I(0) = N_e^2)")
    p.add_argument("--out", type=str, default=None, help="Output .txt file (columns: q  I(q)). Defaults to stdout.")
    args = p.parse_args(argv)

    pos, labels = read_xyz(args.xyz)
    q = q_grid_f90(nq=args.nq, qmin=args.qmin, qmax=args.qmax)

    normalize = not args.no_normalize
    ion_map = json.loads(args.ion_map) if args.ion_map else None
    cm = load_cm_table(args.affl) if args.backend == "affl" else None
    if args.j2:
        if args.inelastic:
            p.error("--inelastic cannot be combined with --j2 (the j2 component is purely elastic)")
        Iq = intensity_j2_xray(pos, labels, q, cm, backend=args.backend, ion_map=ion_map, normalize=normalize)
        columns, header = [q, Iq], f"q[1/Å]    I_j2(q) backend={args.backend}"
    elif args.inelastic:
        I_el, I_inel = intensity_molecular_xray(pos, labels, q, cm, backend=args.backend, ion_map=ion_map,
                                                inelastic=args.inelastic, normalize=normalize)
        columns = [q, I_el, I_inel, I_el + I_inel]
        header = f"q[1/Å]    I_el(q)    I_inel(q)    I_tot(q) backend={args.backend} inelastic={args.inelastic}"
    else:
        Iq = intensity_molecular_xray(pos, labels, q, cm, backend=args.backend, ion_map=ion_map, normalize=normalize)
        columns, header = [q, Iq], f"q[1/Å]    I(q) backend={args.backend}"

    if args.out:
        np.savetxt(args.out, np.column_stack(columns), header=header)
    else:
        for row in zip(*columns):
            print(f"{row[0]:12.6f}  " + "  ".join(f"{v:16.8f}" for v in row[1:]))

if __name__ == "__main__":
    main()
