import numpy as np
import os
import sys
import pandas as pd
from Bio import PDB
import argparse
import warnings
warnings.filterwarnings("ignore")
#python get_partzone.py -i /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z -g /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_pro_chain_GPsite_score_aa -o /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z

def get_split_range(domain):
    domain_range = []
    for d in domain.split('_'):
        if '-' in d:
            d1 = int(d.split('-')[0])
            d2 = int(d.split('-')[1])+1
            for i in range(d1, d2):
                domain_range.append(i)
        else:
            domain_range.append(int(d))
        
    return domain_range

# find top/2 aa ---> residue_list  60分
def pdb_top_resi(pdb_path, pdb_list, Chainsaw_out_path, split_domain_length):
    pdb_top_resi_dict = {}
    df = pd.read_csv(Chainsaw_out_path, sep='\t', header=0, index_col='chain_id')
    for pdb in sorted(pdb_list):
        print(pdb)
        if list(df.loc[[pdb]]['nres'])[0]>=split_domain_length and list(df.loc[[pdb]]['chopping'])[0] != 'NULL' and pd.isna(list(df.loc[[pdb]]['chopping'])[0]) == False:
            print('------------need domain split--------------')
            domain_str = list(df.loc[[pdb]]['chopping'])[0]
            pdb_file = os.path.join(pdb_path, pdb+'.pdb')
            updated_domain_str = update_domain_info_main(pdb_file, domain_str)
            print('updated_domain_str',updated_domain_str)
            domain_split = updated_domain_str.split(',')  #14-149_290-412,152-287_422-450
            for i in range(len(domain_split)):
                domain = domain_split[i]
                domain_range = get_split_range(domain)
                pdb_top_resi_dict[pdb+'_domain'+str(i)] = domain_range

        else:
            pdb_top_resi_dict[pdb] = [i for i in range(1, list(df.loc[[pdb]]['nres'])[0]+1)]
            
    return pdb_top_resi_dict
    

def resi_2_coord(pdb_path, pdb_top_resi_dict):
    pdb_top_coord_dict = {}
    for key in pdb_top_resi_dict.keys():   # 1ACB_I
        parser = PDB.PDBParser()
        io = PDB.PDBIO()
        print(key)
        pdb_c = key.split('_domain')[0]
        pdb_file = os.path.join(pdb_path, pdb_c + '.pdb')
        chain_id = pdb_c.split('_')[-1]
        residue_list = pdb_top_resi_dict[key]
        struct = parser.get_structure('A',pdb_file)
        
        target_coords = []
        for model in struct:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.id[1] in residue_list:
                            for atom in residue:
                                atom_coord_array = np.array(atom.get_coord())
                                target_coords.append(atom_coord_array)
                                
        pdb_top_coord_dict[key] = target_coords
        
    return pdb_top_coord_dict


def read_ply(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Extract the header
    header = []
    vertex_count = 0
    face_count = 0
    is_vertex_section = False
    is_face_section = False
    vertices = []
    faces = []
    
    for line in lines:
        if line.startswith("end_header"):
            header.append(line)
            break
        header.append(line)
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
    
    # Extract the vertex and face data
    for line in lines[len(header):]:
        if vertex_count > 0:
            vertices.append(line.strip().split())
            vertex_count -= 1
        elif face_count > 0:
            faces.append(line.strip().split())
            face_count -= 1
    
    return header, np.array(vertices, dtype=float), np.array(faces, dtype=int)

def write_ply(filename, header, vertices, faces):
    with open(filename, 'w') as f:
        for line in header:
            if line.startswith("element vertex"):
                f.write(f"element vertex {len(vertices)}\n")
            elif line.startswith("element face"):
                f.write(f"element face {len(faces)}\n")
            else:
                f.write(line)
        for vertex in vertices:
            f.write(" ".join(map(str, vertex)) + "\n")
        for face in faces:
            f.write("3 " + " ".join(map(str, face)) + "\n")

def filter_vertices(vertices, faces, target_coords, threshold):
    filtered_vertices = []
    filtered_indices = []
    
    for i, vertex in enumerate(vertices):
        min_dist = np.min([np.linalg.norm(vertex[:3] - coord) for coord in target_coords])
        if min_dist <= threshold:
            filtered_vertices.append(vertex)
            filtered_indices.append(i)
    
    filtered_indices_map = {old_idx: new_idx for new_idx, old_idx in enumerate(filtered_indices)}
    
    filtered_faces = []
    for face in faces:
        if all(idx in filtered_indices_map for idx in face[1:]):
            filtered_faces.append([filtered_indices_map[idx] for idx in face[1:]])
    
    return np.array(filtered_vertices), np.array(filtered_faces)


def get_partzone_ply(pdb_path, target_coords_dict, out_path, threshold = 5.0):
    
    for key in target_coords_dict.keys():
        print(key)
        pdb_c = key.split('_domain')[0]
        input_ply = os.path.join(pdb_path, pdb_c + '_all_5.0.ply')
        if len(key.split('_domain')) == 1:
            output_ply = os.path.join(pdb_path, pdb_c + '_all_5.0_filtered.ply')
        else:
            output_ply = os.path.join(pdb_path, pdb_c + '_all_5.0_filtered_domain_'+key.split('_domain')[-1]+'.ply')
        target_coords = target_coords_dict[key]
        
        header, vertices, faces = read_ply(input_ply)
    
        filtered_vertices, filtered_faces = filter_vertices(vertices, faces, target_coords, threshold)
    
        write_ply(output_ply, header, filtered_vertices, filtered_faces)    
    

# check the cloud point number
def count_lines_between(file_path, start_marker, end_marker):
    with open(file_path, 'r') as file:
        # 初始化变量
        found_start_marker = False
        line_count = 0
        start_line_number = None
        end_line_number = None

        # 遍历文件中的每一行
        for line_number, line in enumerate(file, start=1):
            # 检查是否找到了开始标记
            if start_marker in line:
                found_start_marker = True
                start_line_number = line_number
                continue  # Skip the start_marker line itself
            
            # 如果已经找到开始标记，继续检查
            if found_start_marker:
                if line.startswith(end_marker):
                    end_line_number = line_number
                    break  # 结束遍历，因为已经找到了结束标记

        if start_line_number is not None and end_line_number is not None:
            return end_line_number - start_line_number
        else:
            return None  # 如果没有找到标记，返回 None 表示未找到
            

def get_files_and_folders(directory):
    files_and_folders = os.listdir(directory)
    return files_and_folders

def delete_files_with_prefix(directory, prefix):
    files_and_folders = get_files_and_folders(directory)
    for item in files_and_folders:
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path):
            if item.startswith(prefix):
                os.remove(full_path)
                
def check_type(k):
    type_dict = {'0':'Protein', '1':'DNA'}
    if k not in list(type_dict.keys()):
        print("Undefined Interaction Type!")
        sys.exit(1)
    return type_dict[k]
    
############ Set to your own path! ############
#pdb_path = out_path = whole_ply_path = '/home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z'
#GPsite_pro_path = '/home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_pro_chain_GPsite_score_aa/surface_HS'
#pdb_list = []
#filter_ply_path = '/home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z'
########################


def format_tuples_list(tuples_list):
    # 将每个元组转换为 "a-b" 格式，然后用 "_" 连接
    return "_".join(f"{a}-{b}" for a, b in tuples_list)

def format_list(lst):
    if not lst:
        return ""
    result = []
    start = lst[0]
    prev = lst[0]
    for num in lst[1:]:
        if num != prev + 1:
            if start == prev:
                result.append(str(start))
            else:
                result.append(f"{start}-{prev}")
            start = num
        prev = num
    if start == prev:
        result.append(str(start))
    else:
        result.append(f"{start}-{prev}")
    return "_".join(result)


def read_pdb_coordinates(pdb_file):
    # 解析PDB文件，返回每个氨基酸的Cα原子坐标（残基编号和坐标）
    parser = PDB.PPBuilder()
    structure = PDB.PDBParser(QUIET=True).get_structure("protein", pdb_file)
    amino_acids = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue):  # 判断是否为氨基酸
                    for atom in residue:
                        if atom.get_id()=='CA':
                            amino_acids.append((residue.get_id()[1], atom.get_coord()))  # 获取残基编号和坐标
    return amino_acids

def parse_domain(domain_str):
    # 解析domain字符，将其转换为一个包含所有区间的list，每个domain是一个由多个氨基酸范围（start, end）组成的元组
    domains = []
    domain_parts = domain_str.split(',')
    
    for part in domain_parts:
        domain_ranges = []
        ranges = part.split('_')
        for range_str in ranges:
            start, end = map(int, range_str.split('-'))
            domain_ranges.append((start, end))
        domains.append(domain_ranges)
    return domains

def compute_distance(coord1, coord2):
    # 计算两个氨基酸之间的欧式距离
    return np.linalg.norm(coord1 - coord2)

def assign_undomain_amino_acids(aa_coordinates, domain_ranges):
    # 将没有分配到任何domain的氨基酸归类到最近的domain
    unassigned = [aa for aa in aa_coordinates if not any(start <= aa[0] <= end for domain in domain_ranges for (start, end) in domain)]
    #print('unassigned', unassigned)
    assignments = []
    for amino_acid in unassigned:
        aa_residue, aa_coord = amino_acid
        closest_domain = None
        min_distance = float('inf')
        
        for domain in domain_ranges:
            for (domain_start, domain_end) in domain:
                # 获取当前domain中所有氨基酸坐标
                domain_coords = [coord for (residue, coord) in aa_coordinates if domain_start <= residue <= domain_end]
                
                # 计算距离
                for coord in domain_coords:
                    distance = compute_distance(aa_coord, coord)
                    if distance < min_distance:
                        min_distance = distance
                        closest_domain = domain
        
        # 将未分配的氨基酸归入最近的domain
        assignments.append((aa_residue, closest_domain))
    #print('assignments', assignments)
    return assignments

def update_domain_info(domain_str, original_domains, assignments):  
    updated_domains = {}
    
    for d in original_domains:
        formatted_d = format_tuples_list(d)
        updated_domains[formatted_d] = []
        
    for a in assignments:
        for d in original_domains:
            if a[1] == d:
                updated_domains[format_tuples_list(d)].append(a[0])
    #print('updated_domains', updated_domains)    
    
    updated_domain_list = []
    for d in domain_str.split(','):
        if format_list(updated_domains[d]) != '':
            updated_domain_list.append(d + '_' + format_list(updated_domains[d]))
        else:
            updated_domain_list.append(d)
    updated_domains_str = ','.join(updated_domain_list)
    return updated_domains_str

def update_domain_info_main(pdb_file, domain_str):
    # 读取PDB文件中的氨基酸坐标
    aa_coordinates = read_pdb_coordinates(pdb_file)
    # 解析已有的domain信息
    domain_ranges = parse_domain(domain_str)
    # 将未分配的氨基酸归类到最近的domain
    assignments = assign_undomain_amino_acids(aa_coordinates, domain_ranges)
    # 更新domain信息
    updated_domain_str = update_domain_info(domain_str, domain_ranges, assignments)
    return updated_domain_str


'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-s','--chainsaw_split', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')   #/data/zaw/software/Chainsaw/NB_target_dup_water_new/structure_split.txt
    parser.add_argument('-o','--output_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    args = parser.parse_args()
    
    pdb_list = []
    pdb_path = out_path = whole_ply_path = args.input_dir
    Chainsaw_out_path = args.chainsaw_split
    filter_ply_path = args.output_dir

    # 去除同源蛋白二聚体的重复文件
    for file in os.listdir(pdb_path):
        if file.endswith('_all_5.0.ply'):  # 1a22_A_all_5.0.ply
            pdb_list.append(file.split('_all_5.0')[0])      # 1a22_A


    print('Total pdb num:', len(pdb_list))

    #get_domain_split(pdb_path, Chainsaw_out_path)

    pdb_top_resi_dict = pdb_top_resi(pdb_path, pdb_list, Chainsaw_out_path)
    pdb_top_coord_dict = resi_2_coord(pdb_path, pdb_top_resi_dict)
    get_partzone_ply(pdb_path, pdb_top_coord_dict, out_path, 5.0)

main()
'''

def NB_surfpart(output_dir, domain_out_tsv, split_domain_length):   
    pdb_list = []
    pdb_path = out_path = whole_ply_path = output_dir
    Chainsaw_out_path = domain_out_tsv
    filter_ply_path = output_dir

    # 去除同源蛋白二聚体的重复文件
    for file in os.listdir(pdb_path):
        if file.endswith('_all_5.0.ply'):  # 1a22_A_all_5.0.ply
            pdb_list.append(file.split('_all_5.0')[0])      # 1a22_A


    print('Total pdb num:', len(pdb_list))

    #get_domain_split(pdb_path, Chainsaw_out_path)

    pdb_top_resi_dict = pdb_top_resi(pdb_path, pdb_list, Chainsaw_out_path, split_domain_length)
    pdb_top_coord_dict = resi_2_coord(pdb_path, pdb_top_resi_dict)
    get_partzone_ply(pdb_path, pdb_top_coord_dict, out_path, 5.0)
