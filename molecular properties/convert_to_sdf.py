import csv
import argparse
import numpy as np
import multiprocessing as mp
from rdkit import Chem
from rdkit.Chem import AllChem

def load_csv1(path, x_name, y_name):
    x, y = [], []

    with open(path) as file:
        reader = csv.DictReader(file)
        for i,row in enumerate(reader):
            # print(row)
            # if i ==50:
            #     break
            x.append(row[x_name])

            for i in range(len(y_name)):
                if row[y_name[i]] != '':
                    y.append(float(row[y_name[i]]))
                else:
                    y.append(None)

    x = np.array(x, dtype=str)
    y = np.array(y, dtype=float)
    # print("cc",x)
    # print("vv",y)
    # print ('xxx', x.shape[0], y.shape[0])
    return x, y
def load_csv(path, x_name, y_name):
    x, y = [], []

    with open(path) as file:
        reader = csv.DictReader(file)
        for row in reader:
            # print(row)
            x.append(row[x_name])
            y.append(float(row[y_name]))

    x = np.array(x, dtype=str)
    y = np.array(y, dtype=float)

    return x, y


def optimize_conformer(idx, smi, m, algo="MMFF"):
    print("Calculating {}: {} ...".format(idx, Chem.MolToSmiles(m)))

    mol = Chem.AddHs(m)

    if algo == "ETKDG":
        # Landrum et al. DOI: 10.1021/acs.jcim.5b00654
        k = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        if k != 0:
            return None, None, None

    elif algo == "UFF":
        # Universal Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=20000)
            idx = np.argmin(arr, axis=0)[1]
            conf = mol.GetConformers()[idx]
            mol.RemoveAllConformers()
            mol.AddConformer(conf)

    elif algo == "MMFF":
        # Merck Molecular Force Field
        AllChem.EmbedMultipleConfs(mol, 50, pruneRmsThresh=0.5)
        try:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
        except ValueError:
            return None, None, None

        if not arr:
            return None, None, None

        else:
            arr = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=20000)

            idx = np.argmin(arr, axis=0)[1]
            print("dfsfgsdgsgsdrghsdfgf")
            # print ((idx))
            conf = mol.GetConformers()[int(idx)]
            # print("ff",conf)
            mol2 = Chem.Mol(mol)
            # mol.RemoveAllConformers()
            # mol.AddConformer(conf)
            mol2.RemoveAllConformers()
            mol2.AddConformer(conf)

    # mol = Chem.RemoveHs(conf)
    mol2 = Chem.RemoveHs(mol2)
    return smi, mol2


def random_rotation_matrix():
    theta = np.random.rand()
    r_x = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand()
    r_y = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand()
    r_z = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def rotate_molecule(path, target_path, count=10):
    # Load dataset
    mols = Chem.SDMolSupplier(path)
    rotated_mols = []

    print("Loaded {} Molecules from {}".format(len(mols), path))

    print("Rotating Molecules...")
    for mol in mols:
        for _ in range(count):
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()

                pos = list(mol.GetConformer().GetAtomPosition(atom_idx))
                pos_rotated = np.matmul(random_rotation_matrix(), pos)

                mol.GetConformer().SetAtomPosition(atom_idx, pos_rotated)

            rotated_mols.append(mol)

    w = Chem.SDWriter(target_path)
    for m in rotated_mols:
        if m is not None:
            w.write(m)
    print("Saved {} Molecules to {}".format(len(rotated_mols), target_path))


def converter(path, target_path, dataset, algo, process=20, debug=False):
    # Load dataset
    print("Loading Dataset...")
    # x_y_name_dict = {'bace': ('mol','Class'), 'esol': ('smiles','measured log solubility in mols per litre'), 'lipop': ('smiles','exp'), 'freesolv': ('smiles','expt')}
    x_y_name_dict = {'tox21':("smiles","NR-AR,NR-AR-LBD,NR-AhR,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53".split(',')),'ggg':("smiles","FDA_APPROVED,CT_TOX".split(',')), 'clintox':("smiles","FDA_APPROVED,CT_TOX".split(','))}
    name, target_name = x_y_name_dict[dataset]
    if ".csv" in path:
        x, y = load_csv1(path, name, target_name)
        times = y.shape[0] / x.shape[0]
        smis, mols = [], []
        props = [[] for i in range(int(times))]
        for i in range(len(x)):
            smi = x[i]
           
            # print("uu",smi)
            # print("yt",prop)
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                smis.append(smi)
                mols.append(mol)
                # props.append(prop)
        for i in range(0, y.shape[0], int(times)):
            for index, j in enumerate(range(i, i + int(times))):
                props[index].append(y[j])

        # print("dgfrdg",props)
        mol_idx = list(range(len(mols)))
        algo = [algo]*len(mols)

    else:
        raise ValueError("Unsupported file type.")
    print("Loaded {} Molecules from {}".format(len(mols), path))

    # Optimize coordinate using multiprocessing
    print("Optimizing Conformers...")
    # pool = mp.Pool(process)
    # smi_list, mol_list= [], []
    # aa=list(zip(mol_idx, smis, mols, algo))
    # for i in range(len(aa)):
    #     smi,mol =optimize_conformer(aa[i][0],aa[i][1],aa[i][2],aa[i][3])
    #     smi_list.append(smi)
    #     mol_list.append(mol)
        # prop_list.append(prop)
    pool = mp.Pool(process)
    results = pool.starmap(optimize_conformer, zip(mol_idx, smis, mols, algo))
    # optimize_conformer(mol_idx, smis, mols, algo)
    # print("ff",(results))
    # Collect results
    smi_list, mol_list = [], []
    for sm in results:
        smi_list.append(sm[0])
        mol_list.append(sm[1])
    #     prop_list.append(prop)

    # Remove None and add properties
    mol_list_filtered = []
    print("ffff",len(smi_list))
    print("ffff",len(mol_list))



    for mol_index, (smi, mol) in enumerate(zip(smi_list, mol_list)):
        print ('wanxuxuxu')
        if mol is not None:
            mol.SetProp("smile", str(smi))
            for index, target in enumerate(target_name):
                mol.SetProp(target , str(props[index][mol_index]))
                
            mol_list_filtered.append(mol)
        
        else:
            print('error')
    # print("wewewew",(mol.GetProp(fn) for fn in target_name ) )
    print("{} Molecules Optimized".format(len(mol_list_filtered)))

    # Save molecules
    print("Saving File...")
    w = Chem.SDWriter(target_path)
    for m in mol_list_filtered:
        w.write(m)
    print("Saved {} Molecules to {}".format(len(mol_list_filtered), target_path))

# run: python convert_to_sdf.py --dataset esol --process 2 --algo MMFF
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=int, default=20)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--algo', type=str, default='MMFF') # ETKDG, UFF, MMFF
    args = parser.parse_args()

    converter("./data/%s.csv"%(args.dataset), "./data/%s.sdf"%(args.dataset), args.dataset, args.algo, process=args.process)
