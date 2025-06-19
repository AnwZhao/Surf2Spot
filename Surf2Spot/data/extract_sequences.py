import os
from Bio import PDB
import argparse

def extract_sequences_from_pdb(pdb_path, output_file):
    aa_dict = {'ALA':'A','CYS':'C', 'HIS':'H', 'ARG':'R', 'LYS':'K',
    'ILE':'I', 'PHE':'F', 'MET':'M',  'LEU':'L',  'PRO':'P',
    'GLY':'G',  'ASN':'N', 'VAL':'V', 'TYR':'Y',
    'GLN':'Q',  'THR':'T',  'SER':'S',   'ASP':'D', 'GLU':'E','TRP':'W'}
    # 创建一个 FASTA 文件并写入
    with open(output_file, 'w') as fasta_file:
        for file in sorted(os.listdir(pdb_path)):
            if file.endswith('.pdb'):
                pdb_file = os.path.join(pdb_path, file)
                # 解析 PDB 文件
                parser = PDB.PDBParser(QUIET=True)
                structure = parser.get_structure(os.path.basename(pdb_file).split('.')[0], pdb_file)
            
                for model in structure:
                    for chain in model:
                        chain_id = chain.get_id()
                        # 过滤掉链编号为 'Z' 的链
                        if chain_id != 'Z':
                            # 提取序列
                            sequence = ''.join(aa_dict[residue.resname] for residue in chain if PDB.is_aa(residue))
                            # 转换成 FASTA 格式并写入文件
                            fasta_file.write(f">{file.split('.')[0].split('_')[0]}_{file.split('.')[0].split('_')[1]}\n")
                            fasta_file.write(f"{sequence}\n")
                            print(file.split('_all_')[0],'DONE')
    fasta_file.close()



