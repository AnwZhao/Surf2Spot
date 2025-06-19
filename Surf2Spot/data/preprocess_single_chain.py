from Bio import PDB
import os
import argparse
from pymol import cmd, stored
import pandas as pd
from biopandas.pdb import PandasPdb
import warnings
warnings.filterwarnings("ignore")

def get_chain_length(pdb_file, chain_id):
    # 创建PDB解析器对象
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    
    # 获取第一个模型
    model = structure[0]
    
    # 获取指定链
    chain = model[chain_id]
    
    # 返回链的氨基酸数量
    return len(list(chain.get_residues()))

def remove_HOH(input_pdb, output_pdb):  
    cmd.reinitialize()
    
    # 加载原始 PDB 文件
    cmd.load(input_pdb)  # 替换为您的 PDB 文件名
    # 去除水分子
    cmd.remove('resn HOH')
    cmd.save(output_pdb, 'all')


def parse_pdb_file(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    dna_chains = set()
    protein_chains = set()

    # Define amino acid residues (for simplicity, consider standard 20 amino acids)
    amino_acids = set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY',
                       'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
                       'THR', 'TRP', 'TYR', 'VAL'])

    for model in structure:
        for chain in model:
            is_protein = False
            for residue in chain:
                if residue.get_resname() in amino_acids:
                    is_protein = True
                    break

            if is_protein:
                protein_chains.add(chain.get_id())
            else:
                if get_chain_length(pdb_file, chain.get_id())>=10:
                    dna_chains.add(chain.get_id())

    return list(dna_chains), list(protein_chains)

def get_inter_chain_pdb(input_pdb, chain_pro, output_pdb):  
    cmd.reinitialize()
    
    # 加载原始 PDB 文件
    cmd.load(input_pdb)  # 替换为您的 PDB 文件名
    # 去除水分子
    cmd.select('get',f'(resn ala+arg+asn+asp+cys+glu+gln+gly+his+ile+leu+lys+met+phe+pro+ser+thr+trp+tyr+val) and chain {chain_pro}')
    cmd.save(output_pdb, 'get')


def check_aa_in_selection(input_pdb, chain_id):
    cmd.reinitialize()
    cmd.load(input_pdb)  # 替换为您的 PDB 文件名
    cmd.select(chain_id,f'(resn ala+arg+asn+asp+cys+glu+gln+gly+his+ile+leu+lys+met+phe+pro+ser+thr+trp+tyr+val) and chain {chain_id}')
    # 获取selection中的所有原子模型
    atom_num = cmd.count_atoms(chain_id)
    if atom_num > 0:
        return True
    else:
        return False
    

    
#renum_resi_id
def renum_pdb_resi_num(input_pdb, pro_chain, output_pdb):
    pdb = PandasPdb().read_pdb(input_pdb)
    df = pdb.df['ATOM']
    chain_df = df[df['chain_id'] == pro_chain]
    resi_number_list = list(chain_df['residue_number'])
    resi_number_list_nodup = sorted(list(set(resi_number_list)))
    new_resi_number_list = []
    for j in range(1,len(resi_number_list_nodup)+1):
        resi = resi_number_list_nodup[j-1]
        new_resi_number_list += [j]*resi_number_list.count(resi)
        
    chain_df['residue_number'] = new_resi_number_list
    renum_df = chain_df
    #print(renum_df)
    
    filename = 'empty.pdb'  
    try:  
        with open(filename, 'x') as file:  
            pass  # 同样，这里什么都不做  
    except FileExistsError:  
         print(f"文件 '{filename}' 已经存在，因此没有创建新文件。")
    pdb = PandasPdb().read_pdb(filename)
    pdb.df['ATOM'] = renum_df
    pdb.to_pdb(output_pdb, append_newline=True)
    os.remove(filename)


def get_chain_ids(pdb_file):
    # 使用 biopandas 读取 PDB 文件
    pdb = PandasPdb().read_pdb(pdb_file)
    # 提取所有链ID，并去重
    df = pdb.df['ATOM']
    chain_ids = list(set(list(df['chain_id'])))
    # 返回排序后的链ID
    return sorted(chain_ids)
    

'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o','--out_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    args = parser.parse_args()

    input_path = args.input_dir
    output_path_pro = args.out_dir

    if not os.path.exists(output_path_pro):
        os.makedirs(output_path_pro)
        

    for f in os.listdir(input_path):
        if f.endswith('.pdb'):
            pdb_file = os.path.join(input_path, f)
            chain_list = get_chain_ids(pdb_file)
            print(f)
            for c in chain_list:
                pdb_c = f.split('.')[0]
                pdb = f.split('.')[0]
                if check_aa_in_selection(pdb_file, c):
                    input_pdb = pdb_file
                    output_pdb = os.path.join(output_path_pro, pdb+f'_{c}.pdb')
                    get_inter_chain_pdb(input_pdb, c, output_pdb)
                    remove_HOH(output_pdb, output_pdb)
                    renum_pdb_resi_num(output_pdb,c,output_pdb)
                    print('get protein_chain', c)
            
main()
'''

def get_single_chain(input_dir, output_dir):
    input_path = input_dir
    output_path_pro = output_dir

    if not os.path.exists(output_path_pro):
        os.makedirs(output_path_pro)
 
    for f in os.listdir(input_path):
        if f.endswith('.pdb'):
            pdb_file = os.path.join(input_path, f)
            chain_list = get_chain_ids(pdb_file)
            print(f)
            for c in chain_list:
                pdb_c = f.split('.')[0]
                pdb = f.split('.')[0]
                if check_aa_in_selection(pdb_file, c):
                    input_pdb = pdb_file
                    output_pdb = os.path.join(output_path_pro, pdb+f'_{c}.pdb')
                    get_inter_chain_pdb(input_pdb, c, output_pdb)
                    remove_HOH(output_pdb, output_pdb)
                    renum_pdb_resi_num(output_pdb,c,output_pdb)
                    print('get protein_chain', c)
            
    