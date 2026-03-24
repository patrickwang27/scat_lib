from pyscf import gto, scf, cc
import numpy as npp
import pyscf.tools
import lzma
from scipy.io import FortranFile
from pyscf import lib
from pyscf.geomopt.geometric_solver import optimize
import pyscf.cc.ccsd_rdm as ff
import numpy as np
import os
import mo2ao
import time 
from pyscf.scf import addons
#nth=lib.num_threads()
#print(nth)
#lib.num_threads(1)
#print(lib.num_threads())
def mo2ao_2rdm_split(dm2_mo, C):
    """
    Transform a 2-RDM Γ(p,q,r,s) from MO basis to AO basis:

        Γ_AO(μ,ν,λ,σ) = Σ_pqrs C_{μp} C_{νq} C_{λr} C_{σs} Γ_MO(p,q,r,s)

    Split into 4 contractions for speed and memory stability.
    """
    # Step 1: contract first MO index p → AO index μ
    # X(μ,q,r,s) = Σ_p C(μ,p) * Γ(p,q,r,s)
    X = np.einsum("mp,pqrs->mqrs", C, dm2_mo, optimize=True)

    # Step 2: contract second index q → AO index ν
    # Y(μ,ν,r,s) = Σ_q C(ν,q) * X(μ,q,r,s)
    Y = np.einsum("nq,mqrs->mnrs", C, X, optimize=True)

    # Step 3: contract third index r → AO index λ
    # Z(μ,ν,λ,s) = Σ_r C(λ,r) * Y(μ,ν,r,s)
    Z = np.einsum("lr,mnrs->mnl s", C, Y, optimize=True)

    # Step 4: contract final index s → AO index σ
    # Γ_AO(μ,ν,λ,σ) = Σ_s C(σ,s) * Z(μ,ν,λ,s)
    Gamma_AO = np.einsum("ks,mnls->mn lk", C, Z, optimize=True)

    return Gamma_AO

time1=time.time()
mol = gto.M(
    atom='Ne',    
    basis='aug-cc-pv5z',
    spin=0,
    charge=0, 
    unit='A',
    cart='True',
    symmetry=False
)



mol.unit = 'A'
mol.build()

mf = scf.RHF(mol)
mf.run()  # this is UHF
print(np.shape(mf.mo_coeff))
pyscf.tools.molden.from_scf(mf, 'X.mld', ignore_h=True)
#First routine RUn_ROHF(INPUT='CHD.XYZ')
mycc = cc.ccsd.CCSD(mf).set(max_cycle=200).run()  # this is UCCSD
#dm2 = mycc.make_rdm2()


# AO "trace" with overlap


Gamma=mycc.make_rdm2(ao_repr=False)
gamma=mycc.make_rdm1(ao_repr=False)

# gamma: (n,n), Gamma: (n,n,n,n)
Gamma_H = np.einsum("pq,rs->pqrs", gamma, gamma)      # direct
Gamma_X = np.einsum("ps,qr->pqrs", gamma, gamma)      # exchange
Gamma_mf = Gamma_H - 0.5*Gamma_X                          # Grassmann product
lambda_corr = Gamma - Gamma_mf                        # cumulant part
print("||Gamma - (Gamma_mf + lambda)||_F =",
      np.linalg.norm(Gamma - (Gamma_mf + lambda_corr)))
trace = np.einsum("pqpq->", Gamma_H)
print('trace of H matrix: ', trace)
# 2) Cumulant has zero 1-body contractions
lam_pr = np.einsum("pqrs->pr", lambda_corr)          # sum over q,s
lam_qs = np.einsum("pqrs->qs", lambda_corr)          # sum over p,r
print("||Tr_2(lambda)||_F over (q,s) =", np.linalg.norm(lam_pr))
print("||Tr_2(lambda)||_F over (p,r) =", np.linalg.norm(lam_qs))

# 3) Mean-field part gives (N-1) * gamma when traced over 2nd electron
N = mol.nelectron
Gamma_mf_contr = np.einsum("pqrs->pq", Gamma_mf)     # sum over r,s
print("||Gamma_mf_contr - (N-1)*gamma||_F =",
      np.linalg.norm(Gamma_mf_contr - (N-1)*gamma))
print("transformation done")
#Gamma.tofile('twordm.datb')
nmo=np.size(Gamma_H[:,0,0,0])
N = mol.nelectron

print("Sum Γ_ij,ij AO =", np.einsum('iijj->', Gamma), " expected ", N*(N-1))
mo2ao.create_Zcotr(mf,mol,Gamma)
os.system('python3.8 /u/ajmk/chem1721/File_creator_Zcotr/File_creator_molden.py Ne_av5z_ccsd.dat')
#os.system('python3.8 /u/ajmk/chem1721/XSCAT-Eirik/src/File_creator_molden_pyscf.py')
os.system("/u/ajmk/chem1721/PyXSCAT_last_commit/PyXSCAT/src/Main2.exe")
