import numpy as np
import os
import sys
import pandas as pd
from Bio import PDB
import argparse
import warnings
warnings.filterwarnings("ignore")
#python get_partzone.py -i /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z -g /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_pro_chain_GPsite_score_aa -o /home/anwzhao/my_development/HSFilter/GNN-M_data_DNA/all_data/gpsite_DNA_protein_Z

# find top/2 aa ---> residue_list  60分
def pdb_top_resi(pdb_list, ppi_type, GPsite_pro_path, rate):
    pdb_top_resi_dict = {}
    for pdb in sorted(pdb_list):
        print(pdb)
        df = pd.read_csv(os.path.join(GPsite_pro_path, pdb, 'GPSite_score.csv') )
        if df.empty:
            os.remove(os.path.join(pdb_path, pdb +'.pdb'))
            os.remove(os.path.join(pdb_path, pdb +'_all_5.0.ply'))
        else:
            col_name = ppi_type+'_binding'
            top_score = sorted(list(df[col_name]))[-1]*rate
            
            if len(df[df[col_name] >= top_score])<30:
                top_score = np.percentile(list(df[col_name]), 70)
            if len(df[df[col_name] >= top_score])<30:
                top_score = np.percentile(list(df[col_name]), 70)
               
            df = df[df[col_name] >= top_score]
            residue_id_top2 = list(df['No'])
            pdb_top_resi_dict[pdb] = residue_id_top2
            
    return pdb_top_resi_dict

def resi_2_coord(pdb_path, pdb_top_resi_dict):
    pdb_top_coord_dict = {}
    for key in pdb_top_resi_dict.keys():   # 1ACB_I
        parser = PDB.PDBParser()
        io = PDB.PDBIO()
        pdb_file = os.path.join(pdb_path, key + '.pdb')
        chain_id = key.split('_')[-1]
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
        input_ply = os.path.join(pdb_path, key + '_all_5.0.ply')
        output_ply = os.path.join(pdb_path, key + '_all_5.0_filtered.ply')
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
                
'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-g','--gpsite_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o','--output_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    args = parser.parse_args()
    
    pdb_list = []
    pdb_path = out_path = whole_ply_path = args.input_dir
    GPsite_pro_path = os.path.join(args.gpsite_dir, 'surface_HS')
    filter_ply_path = args.output_dir

    # 去除同源蛋白二聚体的重复文件
    for file in os.listdir(pdb_path):
        if file.endswith('_all_5.0.ply'):  # 1a22_A_all_5.0.ply
            if file.split('_all_5.0')[0] in os.listdir(GPsite_pro_path) and os.path.exists(os.path.join(pdb_path, file.split('.ply')[0]+'_filtered.ply'))==False:
                pdb_list.append(file.split('_all_5.0')[0])      # 1a22_A


    print('Total pdb num:', len(pdb_list))

    pdb_top_resi_dict = pdb_top_resi(pdb_list, 'Protein', GPsite_pro_path)
    pdb_top_coord_dict = resi_2_coord(pdb_path, pdb_top_resi_dict)
    get_partzone_ply(pdb_path, pdb_top_coord_dict, out_path, 5.0)

main()
'''


def HS_surfpart(input_dir, gpsite_dir, rate):    
    pdb_list = []
    pdb_path = out_path = whole_ply_path = input_dir
    GPsite_pro_path = os.path.join(gpsite_dir, 'surface_HS')
    filter_ply_path = input_dir

    # 去除同源蛋白二聚体的重复文件
    for file in os.listdir(pdb_path):
        if file.endswith('_all_5.0.ply'):  # 1a22_A_all_5.0.ply
            if file.split('_all_5.0')[0] in os.listdir(GPsite_pro_path) and os.path.exists(os.path.join(pdb_path, file.split('.ply')[0]+'_filtered.ply'))==False:
                pdb_list.append(file.split('_all_5.0')[0])      # 1a22_A


    print('Total pdb num:', len(pdb_list))

    pdb_top_resi_dict = pdb_top_resi(pdb_list, 'Protein', GPsite_pro_path, rate)
    pdb_top_coord_dict = resi_2_coord(pdb_path, pdb_top_resi_dict)
    get_partzone_ply(pdb_path, pdb_top_coord_dict, out_path, 5.0)
    