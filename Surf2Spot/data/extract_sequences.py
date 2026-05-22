import os
from Bio import PDB
import argparse
import re
import glob

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


def process_esmpdb(folder_path, dry_run=False):
    pattern = re.compile(r'^fold_(.+)_after_model_([0-4])_A\.pdb$')
    all_files = glob.glob(os.path.join(folder_path, "fold_*_after_model_*_A.pdb"))

    tag_dict = {}  # {tag: {'zero': path_or_None, 'others': [paths]}}
    for file_path in all_files:
        basename = os.path.basename(file_path)
        match = pattern.match(basename)
        if not match:
            continue 
        tag, num_str = match.groups()
        num = int(num_str)
        if tag not in tag_dict:
            tag_dict[tag] = {'zero': None, 'others': []}
        if num == 0:
            tag_dict[tag]['zero'] = file_path
        else:
            tag_dict[tag]['others'].append(file_path)

    results = {}
    for tag, info in tag_dict.items():
        zero_file = info['zero']
        others = info['others']
        target_path = os.path.join(folder_path, f"{tag}.pdb")
        if os.path.exists(target_path) and not os.path.samefile(target_path, zero_file):
            os.remove(target_path)

        os.rename(zero_file, target_path)
        for f in others:
            os.remove(f)

