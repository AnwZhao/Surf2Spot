import os
import csv
import argparse
import pandas as pd
from Bio import PDB
from Bio.PDB.DSSP import DSSP
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
# python get_atom_onehot.py -i /home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_GPsite_score_aa -o /home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_Z_most_chain

maxASA = {
    'A': 129.0,  # Alanine
    'R': 274.0,  # Arginine
    'N': 195.0,  # Asparagine
    'D': 193.0,  # Aspartic acid
    'C': 167.0,  # Cysteine
    'Q': 225.0,  # Glutamine
    'E': 223.0,  # Glutamic acid
    'G': 104.0,  # Glycine
    'H': 224.0,  # Histidine
    'I': 197.0,  # Isoleucine
    'L': 201.0,  # Leucine
    'K': 236.0,  # Lysine
    'M': 224.0,  # Methionine
    'F': 240.0,  # Phenylalanine
    'P': 159.0,  # Proline
    'S': 155.0,  # Serine
    'T': 172.0,  # Threonine
    'W': 285.0,  # Tryptophan
    'Y': 263.0,  # Tyrosine
    'V': 174.0   # Valine
}

def extract_atom_coordinates(pdb_file):
    """
    从PDB文件中提取每个氨基酸残基的原子坐标，并返回一个列表。
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    atom_coords = []
    
    # 计算DSSP
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    
    # 获取标准化的溶剂可及表面积
    normalized_asa = []
    for key in dssp.keys():
        res_type = dssp[key][1]
        asa = dssp[key][3]
        max_asa_value = maxASA.get(res_type, None)
        if max_asa_value:
            normalized_asa.append(asa)
    i = 0
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    for atom in residue:
                        atom_coords.append({
                            'residue_name': residue.get_resname(),
                            'residue_id': residue.id[1],
                            'chain_id': chain.id,
                            'atom_name': atom.get_name(),
                            'atom_x': atom.coord[0],
                            'atom_y': atom.coord[1],
                            'atom_z': atom.coord[2],
                            'accessibility':normalized_asa[i],
                            'b_factor':atom.get_bfactor()
                        })
                    i += 1

    return atom_coords

def extract_dssp_info(dssp_file):
    """
    解析DSSP文件，提取每个氨基酸的相关信息，并返回一个字典。
    """
    dssp_info = {}
    with open(dssp_file, 'r') as file:
        lines = file.readlines()
        start_reading = False
        for line in lines:
            if start_reading:
                if len(line) < 120:
                    continue
                if line[5:10].strip() != '':
                    residue_id = int(line[5:10].strip())
                    chain_id = line[11].strip()
                    aa = line[13].strip()
                    structure = line[16].strip()
                    accessibility = int(line[35:38].strip())
                    phi = float(line[103:109].strip())
                    psi = float(line[109:115].strip())
                    dssp_info[(chain_id, residue_id)] = {
                        'aa': aa,
                        'structure': structure,
                        #'accessibility': accessibility,
                        'phi': phi,
                        'psi': psi
                    }
            if line.startswith("  #  RESIDUE AA STRUCTURE"):
                start_reading = True
    return dssp_info
'''
def load_binding_data(binding_file):
    """
    读取binding信息的CSV文件并返回一个字典。
    """
    df = pd.read_csv(binding_file)
    binding_info = df.set_index('No').to_dict('index')
    return binding_info
'''
def save_combined_info_to_csv(atom_coords, dssp_info, output_file):
    """
    将原子坐标、DSSP信息和binding信息结合后保存到CSV文件。
    """
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'X', 'Y', 'Z', 'Residue', 'Residue ID', 'Chain ID', 'Atom', 'AA', 'Structure', 'Accessibility', 'Phi', 'Psi','b_factor'
            #,
            #'Peptide_binding', 'Protein_binding'
        ])
        for atom in atom_coords:
            chain_id = atom['chain_id']
            residue_id = atom['residue_id']
            if (chain_id, residue_id) in dssp_info:
                dssp = dssp_info[(chain_id, residue_id)]
                #binding = binding_info.get(residue_id, {'Peptide_binding': 0, 'Protein_binding': 0})
                writer.writerow([
                    atom['atom_x'], atom['atom_y'], atom['atom_z'],
                    atom['residue_name'], residue_id, chain_id, atom['atom_name'],
                    dssp['aa'], dssp['structure'], atom['accessibility'],
                    dssp['phi'], dssp['psi'],atom['b_factor']
                    #,
                    #binding['Peptide_binding'], binding['Protein_binding']
                ])

def run(pdb_file, dssp_file, output_file):
    atom_coords = extract_atom_coordinates(pdb_file)
    dssp_info = extract_dssp_info(dssp_file)
    save_combined_info_to_csv(atom_coords, dssp_info, output_file)
    print(f"Combined information saved to {output_file}")


def to_one_hot(input_file,output_file):
    # 读取CSV文件
    #input_file = 'combined_info.csv'
    df = pd.read_csv(input_file)

    # 提取氨基酸类别列
    aa_column = df[['AA']]

    # 初始化OneHotEncoder并指定所有可能的类别
    aa_all_categories = ['A', 'C', 'D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']  # 总的类别列表
    encoder = OneHotEncoder(categories=[aa_all_categories], sparse_output=False)

    # 对氨基酸类别进行one-hot编码
    onehot_encoded = encoder.fit_transform(aa_column)

    # 将one-hot编码结果转换为DataFrame并添加列名
    onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out(['AA']))

    # 将one-hot编码后的DataFrame与原始DataFrame合并
    df_onehot = pd.concat([df, onehot_df], axis=1)

    # 删除原始的氨基酸类别列
    df_onehot.drop(columns=['AA'], inplace=True)


    df = df_onehot


    # 提取氨基酸类别列
    st_column = df[['Structure']]

    st_column.fillna('blank',inplace=True)

    # 初始化OneHotEncoder并指定所有可能的类别
    st_all_categories = ['H', 'B', 'E','G','I','S','T','blank']  # 总的类别列表
    encoder = OneHotEncoder(categories=[st_all_categories], sparse_output=False)

    # 对氨基酸类别进行one-hot编码
    onehot_encoded = encoder.fit_transform(st_column)

    # 将one-hot编码结果转换为DataFrame并添加列名
    onehot_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out(['Structure']))

    # 将one-hot编码后的DataFrame与原始DataFrame合并
    df_onehot = pd.concat([df, onehot_df], axis=1)

    # 删除原始的氨基酸类别列
    df_onehot.drop(columns=['Structure'], inplace=True)

    df_onehot.drop(columns=['Residue', 'Residue ID', 'Chain ID', 'Atom'], inplace=True)
    #print(df_onehot.dtypes)
    df_onehot.fillna(0,inplace=True)

    # 保存结果到新的CSV文件
    #output_file = 'combined_info_onehot.csv'
    df_onehot.to_csv(output_file, index=False)

    print(f"One-hot encoded information saved to {output_file}")


'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument("-i",'--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')         
    parser.add_argument("-o", '--out_dir', type=str, metavar='',
                        help='Directory of output files.(Please enter an absolute path)', required=True)
    args = parser.parse_args()

    out2_path = args.out_dir  

    for ply in os.listdir(out2_path):
        #print('ply',ply)
        if ply.endswith('_all_5.0.ply') :
            pdb_name = ply.split('_all_')[0]
            print(pdb_name)
            if not os.path.exists(os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')):
                dssp_file = os.path.join(args.input_dir,pdb_name+'.dssp')
                pdb_file = os.path.join(args.input_dir,pdb_name+'.pdb')
                os.system('mkdssp -i %s -o %s' %(pdb_file,dssp_file))
                output1_file = os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')
                output2_file = os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')
        
                run(pdb_file, dssp_file, output1_file)
                to_one_hot(output1_file,output2_file)

main()
'''

def atom_feature_engineering(output_dir):
    out2_path = output_dir
    for ply in os.listdir(out2_path):
        #print('ply',ply)
        if ply.endswith('_all_5.0.ply') :
            pdb_name = ply.split('_all_')[0]
            print(pdb_name)
            if not os.path.exists(os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')):
                dssp_file = os.path.join(output_dir,pdb_name+'.dssp')
                pdb_file = os.path.join(output_dir,pdb_name+'.pdb')
                os.system('mkdssp -i %s -o %s' %(pdb_file,dssp_file))
                output1_file = os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')
                output2_file = os.path.join(out2_path,pdb_name+'_combined_info_onehot_atom.csv')
        
                run(pdb_file, dssp_file, output1_file)
                to_one_hot(output1_file,output2_file)
