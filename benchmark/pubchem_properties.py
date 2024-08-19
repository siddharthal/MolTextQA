from pypubchem import Compound
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

def get_molecule_properties(pubchem_cid):
    # Retrieve compound information from PubChem
    compound = Compound.from_cid(pubchem_cid)
    
    if compound is None:
        return None
    
    # Get SMILES string
    smiles = compound.canonical_smiles
    
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None
    
    # Retrieve conformers if available
    conformers = compound.record.get("conformers", None)
    
    # If no conformers found, generate 3D coordinates using RDKit
    if not conformers:
        mol_with_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_with_3d, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol_with_3d)
        conformer = mol_with_3d.GetConformer()
        coordinates = conformer.GetPositions().tolist()
    else:
        coordinates = conformers  # If conformers exist, use them

    # Calculate geometric descriptors
    mol_weight = Descriptors.MolWt(mol)
    mol_logp = Descriptors.MolLogP(mol)
    h_bond_donor = rdMolDescriptors.CalcNumHBD(mol)
    h_bond_acceptor = rdMolDescriptors.CalcNumHBA(mol)
    formal_charge = Chem.GetFormalCharge(mol)
    
    # Generate Morgan fingerprint
    morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    return {
        "PubChem_CID": pubchem_cid,
        "IUPAC_Name": compound.iupac_name,
        "SMILES": smiles,
        "Molecular_Formula": compound.molecular_formula,
        "Molecular_Weight": mol_weight,
        "LogP": mol_logp,
        "Hydrogen_Bond_Donors": h_bond_donor,
        "Hydrogen_Bond_Acceptors": h_bond_acceptor,
        "Formal_Charge": formal_charge,
        "Conformers": conformers,
        "3D_Coordinates": coordinates,
        "Morgan_Fingerprint": morgan_fingerprint.ToBitString()
    }


def get_molecule_names(pubchem_cid):
    # Retrieve compound information from PubChem
    compound = Compound.from_cid(pubchem_cid)
    
    if compound is None:
        return None
    
    # Get synonyms (names)
    synonyms = compound.synonyms
    
    # Return synonyms if the list is not empty
    if synonyms:
        return synonyms
    else:
        return "No synonyms available"

if __name__ == "__main__":
    # Example usage
    pubchem_cid_list = [
        2244,  # Aspirin
        7967   # Caffeine
    ]
    
    for cid in pubchem_cid_list:
        properties = get_molecule_properties(cid)
        print(properties)
    
    
    for cid in pubchem_cid_list:
        names = get_molecule_names(cid)
        print(f"PubChem CID {cid} names: {names}")


