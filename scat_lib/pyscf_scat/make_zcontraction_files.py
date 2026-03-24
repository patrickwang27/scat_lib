import numpy as np
from . import molden_reader_nikola_morder as mldreader
import os


types = {'total': '1', 
         'elastic':'2',
         'total_aligned': '3',
         'elastic_aligned' : '4',
         'total_electron' : '5',
         'elastic_electron' : '6',
         'total_j2' : '7',
         'elastic_j2' :'8',
         'resolved_cms' : '9'}

def _make_zcontraction_option(
        mldfile,
        file_name,
        two_rdm_file,
        type='total',
        q_range = (1E-10,250),
        q_points = 1000,
        cutoffcentre = 1E-2,
        cutoffz = 1e-20,
        cutoffmd = 1e-20,
        state1 = 1,
        state2 = 1,
        path= './'):
    '''
    Creates the options.dat file needed for Z-contraction calculations.
    Parameters
    ----------
    mldfile : str
        The path to the Molden file.
    file_name : str
        The name of the output file.
    two_rdm_file : str
        The path to the 2-RDM file.
    type : str, optional
        The type of calculation. Options are 'total', 'elastic', 'inelastic', '
        'total_aligned', 'elastic_aligned', 'inelastic_aligned', 'total_electron',
        'elastic_electron', 'inelastic_electron', 'total_j2'
        'elastic_j2', 'inelastic_j2'. Default is 'total'.
    q_range : tuple, optional
        The range of q values to calculate, in a.u. Default is (1E-10, 250).
    q_points : int, optional
        The number of q points to calculate. Default is 1000.
    cutoffcentre : float, optional
        The cutoff for the Z integral. Default is 0.01.
    cutoffz : float, optional
        The cutoff for the Z integral. Default is 1E-9.
    cutoffmd : float, optional
        The cutoff for the product of the MD coefficients. Default is 1E-20.
    state1 : int, optional
        The first state to consider. Default is 1.
    state2 : int, optional
        The second state to consider. Default is 1.
    path : str, optional
        The path to the directory where the options.dat file will be saved. Default is the current directory.
    Returns
    -------
    None
    
    '''
    
    Nmo_max = 600

    _, atoms, _, _, _, _ = mldreader.read_orbitals(mldfile, N=Nmo_max, decontract=False)

    jeremyR = False
    mcci = False
    hf = False
    molpro = False
    molcas = False
    bagel = False
    geom = atoms.geometry()

    with open(os.path.join(path, 'options.dat'), 'w') as f:
        f.write(str(np.size(atoms.atomic_numbers())) + '\n')
        for i in atoms.atomic_numbers():
            f.write(str(i) + ' ')
        f.write('\n')
        for i in range(np.size(atoms.atomic_numbers())):
            f.write(str(geom[i, :])[1:-1] + '\n')
        f.write(str(cutoffcentre) + '\n')
        f.write(str(cutoffz) + '\n')
        f.write(str(cutoffmd) + '\n')
        f.write(str(jeremyR) + '\n')
        f.write(str(mcci) + '\n')
        f.write(str(hf) + '\n')
        f.write(str(q_range[0]) + ' ' + str(q_range[1]) + ' ' + str(q_points) + ' \n')
        f.write(str(types[type]) + '\n')
        f.write(str(state1) + ' ' + str(state2) + '\n')
        f.write(file_name + '\n')
        f.write(str(molpro) + '\n')
        f.write(str(molcas) + '\n')
        f.write(str(bagel) + '\n')
        f.write('readtwordm' + '\n')
        f.write(two_rdm_file)


def _make_zcontraction_files(mldfile, path='./'):
    """
    Create files needed for Z-contraction from a Molden file.
    """
    Nmo_max = 600
    gtos, atoms, coeffs, mos, groupC,contr = mldreader.read_orbitals(mldfile, N=Nmo_max, decontract=False)
    xx = gtos.x
    yy = gtos.y
    zz = gtos.z
    l = gtos.l
    m = gtos.m
    n = gtos.n
    ga = gtos.ga
    group = gtos.group

    mmod = np.transpose(gtos.mo)

    l = np.asarray(l)
    m = np.asarray(m)
    n = np.asarray(n)
    ga = np.asarray(ga)
    mmod = np.asarray(mmod, dtype=np.float64)



    with open(os.path.join(path, 'basis.dat'), 'w') as f:
        f.write(str(np.size(l)) + '\n')
        for i in range(np.size(l)):
            f.write(str(xx[i]) + ' ' + str(yy[i]) + ' ' + str(zz[i]) + ' ' + str(ga[i]) + ' ' + str(l[i]) + ' ' + str(
                m[i]) + ' ' + str(n[i]) + ' ' + str(group[i]) + ' '+ str(contr[i])+'\n')
    with open(os.path.join(path, 'MOs.dat'), 'w') as f:
        f.write(str(np.size(mmod[:, 0])) + ' ' + str(np.size(mmod[0, :])) + '\n')
        for i in range(np.size(mmod[:, 0])):
            for j in range(np.size(mmod[0, :])):
                f.write(str(mmod[i, j]) + ' ')
            f.write('\n')
    with open(os.path.join(path, 'MOs2.dat'), 'w') as f:
        f.write(str(np.size(mos[:, 0])) + ' ' + str(np.size(mos[0, :])) + '\n')
        for i in range(np.size(mos[:, 0])):
            for j in range(np.size(mos[0, :])):
                f.write(str(mos[i, j]) + ' ')
            f.write('\n')
    with open(os.path.join(path, 'coeffs.dat'), 'w') as f:
        f.write(str(np.size(l)) + '\n')
        for i in range(np.size(l)):
            f.write(str(coeffs[i]) + '\n')
        f.write(str(np.size(groupC)) + '\n')
        count = 1
        for i in range(np.size(groupC)):
            f.write(str(count) + ' ' + str(count + groupC[i] - 1) + ' ' + str(groupC[i]) + '\n')
            count = count + groupC[i]
    return