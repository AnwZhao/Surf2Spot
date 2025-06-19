import numpy as np
from Bio.PDB import PDBParser
from Bio import PDB
from pymol import cmd
import random
import os
import json
import pandas as pd
from sklearn.cluster import DBSCAN
import torch
import warnings
warnings.filterwarnings("ignore")

def remove_duplicates(lst):
    seen = set()
    result = []
    for x in lst:
        while x in seen:
            if random.random() < 0.5:
                x += 0.001
            else:
                x -= 0.001
        seen.add(x)
        result.append(x)
    return result


def extract_coordinates(pdb_file, chain_id):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    coordinates_dict = {}
    aa_num = 0
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    if PDB.is_aa(residue):
                        aa_num += 1
                        res_id = residue.get_id()[1]  # 获取氨基酸编号
                        for atom in residue:
                            coord = tuple(remove_duplicates(atom.coord))  # 获取原子坐标
                            coordinates_dict[coord] = res_id
    return coordinates_dict,aa_num


def find_nearest_point(array1, array2, index):
    # 获取array1中的第index个点
    point = array1[index]
    # 计算array2中每个点与point的距离
    distances = np.linalg.norm(array2 - point, axis=1)
    if min(distances) <= 2:
        # 找到距离最小的点的索引
        nearest_index = np.argmin(distances)
        nearest_point = array2[nearest_index]
        return nearest_point, nearest_index
    else:
        return False


def read_ply_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    vertex_section = False
    xyz_data = []
    hbond_data = []
    for line in lines:
        if line.strip() == "end_header":
            vertex_section = True
            continue
        if vertex_section:
            parts = line.split()
            if len(parts) >= 10:
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                hbond = float(parts[4])
                xyz_data.append([x, y, z])
                hbond_data.append(hbond)
    hbond_data = [-i for i in hbond_data]
    return np.array(xyz_data), hbond_data


def color_pse(pdb_f,out_path,score_list,best_threshold,out_name):
        cmd.reinitialize()
        cmd.delete('all')
        cmd.load(pdb_f)
        pdb = pdb_f.split('/')[-1]
        pdb_name = pdb[:-4]
        print(pdb_name)
        chain_id = pdb_name.split('_')[1]
        cmd.color('palecyan','all')
        cmd.do('set cartoon_transparency, 0.3')
        cmd.do('set dash_gap, 0.2')
        cmd.do('set dash_round_ends, 0')
        cmd.do('set label_size, 18')
        cmd.set('bg_rgb', 'white')
        #cmd.show('surface', 'chain '+ chain_id)
        cmd.color('gray90','chain '+ chain_id)
        choose_resi = []
        for i in range(len(score_list)):
            score = score_list[i]
            g = b = 1 - score
            cmd.set_color(f'color_{i}', [1, g, b])  #color range by score
            #print(1,g,b)
            cmd.color(f"color_{i}", f"{pdb_name} and chain {chain_id} and resi {i+1}")    
            #if score >= 0.5:
            if score >= best_threshold:
                selection = f'name CA and chain {chain_id} and resi {i+1}'
                cmd.label(selection, 'oneletter+resi' )
                cmd.show('sticks',f'chain {chain_id} and resi {i+1}')
                choose_resi.append(i+1)
                
        cmd.save(os.path.join(out_path,f'{out_name}_pre.pse'), pdb_name)
        load_pdb_and_cluster_dynamic(pdb_f, choose_resi,out_path,out_name)

def load_pdb_and_cluster_dynamic(pdb_path, resi_list, out_path, out_name, eps=15.0, min_samples=3):
    """
    从PDB中导入特定残基并进行聚类（不指定聚类数）。

    参数:
        pdb_path (str): PDB文件路径。
        resi_list (list of int): 要分析的残基ID列表。
        eps (float): DBSCAN中邻域的半径参数。
        min_samples (int): DBSCAN中一个簇的最小点数。
        
    返回:
        cluster_results (list of list): 每个聚类包含的残基ID的嵌套列表。
    """
    pdb = pdb_path.split('/')[-1]
    pdb_name = pdb[:-4]
    # 加载PDB文件到PyMOL
    cmd.reinitialize()  # 清空PyMOL会话
    cmd.load(pdb_path, "protein")
    cmd.color('palecyan','all')
    cmd.do('set cartoon_transparency, 0.3')
    cmd.do('set dash_gap, 0.2')
    cmd.do('set dash_round_ends, 0')
    cmd.do('set label_size, 18')
    cmd.set('bg_rgb', 'white')
            
    #cmd.show('surface', 'chain '+ chain_id)
    cmd.color('gray90','all')
    
    # 获取特定残基的坐标
    coordinates = []
    selected_resi = []
    for resi in resi_list:
        selection = f"resi {resi}"
        atom_coords = cmd.get_coords(selection, 1)  # 获取选择的原子坐标
        if atom_coords is not None:
            coordinates.append(atom_coords.mean(axis=0))  # 使用均值代表残基中心
            selected_resi.append(resi)
    
    if not coordinates:
        raise ValueError("未找到任何指定的残基坐标。")
    
    # 转换为numpy数组
    coordinates = np.array(coordinates)
    
    # 使用DBSCAN进行聚类
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
    labels = dbscan.labels_
    
    # 组织聚类结果
    cluster_results = []
    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue  # -1表示噪声点，忽略
        cluster_results.append([selected_resi[i] for i in range(len(labels)) if labels[i] == label])
    
    noise_point = [selected_resi[i] for i in range(len(labels)) if labels[i] == -1]

    for i in range(len(cluster_results)):
        c = cluster_results[i]
        r = round((0.4*(i+1)) % 1,1)
        g = round((0.75*(i+1)) % 1,1)
        b = round((0.9*(i+1)) % 1,1)
        cmd.set_color(f'color_{i}', [r, g, b])
        for ci in c:
            cmd.color(f"color_{i}", f"resi {ci}")
            selection = f'name CA and resi {ci}'
            cmd.label(selection, 'oneletter+resi' )
            cmd.show('sticks',f'resi {ci}')

    cmd.save(os.path.join(out_path,f'{out_name}_cluster.pse'), pdb_name)

def evaluate_with_threshold(aa_score, threshold):
    aa_score_0_1 = []
    for score in aa_score:
        if score >= threshold:
            aa_score_0_1.append(1)
        else:
            aa_score_0_1.append(0)
    return aa_score_0_1


'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('--pdb_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('--ply_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    args = parser.parse_args()
    
    pdb_path = args.pdb_dir
    pre_ply_path = args.ply_dir
    
    best_threshold_list = []
    aa_score_list = []
    aa_score_0_1_list = []

    for pre_ply in os.listdir(pre_ply_path):
        if pre_ply.endswith('.ply'):
            ply_path = os.path.join(pre_ply_path,pre_ply)
            pc_coordinates, pc_score = read_ply_file(ply_path)

            # 示例使用
            if '_domain_' in pre_ply:
                pdb_name = pre_ply.split('_domain_')[0]
                out_name = pdb_name + pre_ply.split('_filtered')[-1].split('.')[0] 
            else:
                pdb_name = pre_ply.split('_pred')[0]
                out_name = pdb_name

            pdb_file = os.path.join(pdb_path,pdb_name+'.pdb')
            chain_id = pdb_name.split('_')[1]  # 替换为你要提取的链编号
            atom_coordinates_dict,aa_num = extract_coordinates(pdb_file, chain_id)
            print('---------pdb_name---------',pre_ply.split('_pred')[0])        
            aa_score = [0]*aa_num        
            aa_score_dup = [0]*aa_num        
            atom_coordinates = []
            for key in atom_coordinates_dict.keys():
                atom_coordinates.append(list(key))
            atom_coordinates = np.array(atom_coordinates)        
            for i in range(len(atom_coordinates)):
                if find_nearest_point(atom_coordinates, pc_coordinates, i)==False:
                    pass
                else:
                    nearest_point_in_pc, nearest_index_in_pc = find_nearest_point(atom_coordinates, pc_coordinates, i)   
                    resi_id = atom_coordinates_dict[tuple(atom_coordinates[i])] #resi_id
                    resi_score = pc_score[nearest_index_in_pc]  #resi_score
                    aa_score[resi_id - 1] += resi_score
                    aa_score_dup[resi_id - 1] += 1
        
            for s in range(len(aa_score)):
                if aa_score_dup[s] != 0:
                    if aa_score_dup[s] >= 1:   #  5  0.420448120312251
                        aa_score[s] = round(aa_score[s]/aa_score_dup[s],3)
                    else:
                        aa_score[s] = 0
            aa_score = torch.Tensor(aa_score).numpy().tolist()
            
            best_threshold = 0.01
            count = len([x for x in aa_score if x > best_threshold])
            if count > 40:
                best_threshold = sorted(aa_score,reverse=True)[39]
                
            aa_score_0_1 = evaluate_with_threshold(aa_score, best_threshold)
            aa_score_list.append(aa_score)
            aa_score_0_1_list.append(aa_score_0_1)    
            color_pse(pdb_file, pre_ply_path, aa_score, best_threshold, out_name)
            df = pd.DataFrame({'aa_id':range(1, len(aa_score)+1),'score':aa_score})
            df.to_csv(os.path.join(pre_ply_path, out_name+'.csv'),sep=',',index=False)
        
main()
'''

def HS_draw(pdb_dir, ply_dir):  
    pdb_path = pdb_dir
    pre_ply_path = ply_dir
    
    best_threshold_list = []
    aa_score_list = []
    aa_score_0_1_list = []

    for pre_ply in os.listdir(pre_ply_path):
        if pre_ply.endswith('.ply'):
            ply_path = os.path.join(pre_ply_path,pre_ply)
            pc_coordinates, pc_score = read_ply_file(ply_path)

            # 示例使用
            if '_domain_' in pre_ply:
                pdb_name = pre_ply.split('_domain_')[0]
                out_name = pdb_name + pre_ply.split('_filtered')[-1].split('.')[0] 
            else:
                pdb_name = pre_ply.split('_pred')[0]
                out_name = pdb_name

            pdb_file = os.path.join(pdb_path,pdb_name+'.pdb')
            chain_id = pdb_name.split('_')[1]  # 替换为你要提取的链编号
            atom_coordinates_dict,aa_num = extract_coordinates(pdb_file, chain_id)
            print('---------pdb_name---------',pre_ply.split('_pred')[0])        
            aa_score = [0]*aa_num        
            aa_score_dup = [0]*aa_num        
            atom_coordinates = []
            for key in atom_coordinates_dict.keys():
                atom_coordinates.append(list(key))
            atom_coordinates = np.array(atom_coordinates)        
            for i in range(len(atom_coordinates)):
                if find_nearest_point(atom_coordinates, pc_coordinates, i)==False:
                    pass
                else:
                    nearest_point_in_pc, nearest_index_in_pc = find_nearest_point(atom_coordinates, pc_coordinates, i)   
                    resi_id = atom_coordinates_dict[tuple(atom_coordinates[i])] #resi_id
                    resi_score = pc_score[nearest_index_in_pc]  #resi_score
                    aa_score[resi_id - 1] += resi_score
                    aa_score_dup[resi_id - 1] += 1
        
            for s in range(len(aa_score)):
                if aa_score_dup[s] != 0:
                    if aa_score_dup[s] >= 1:   #  5  0.420448120312251
                        aa_score[s] = round(aa_score[s]/aa_score_dup[s],3)
                    else:
                        aa_score[s] = 0
            aa_score = torch.Tensor(aa_score).numpy().tolist()
            
            best_threshold = 0.01
            count = len([x for x in aa_score if x > best_threshold])
            if count > 40:
                best_threshold = sorted(aa_score,reverse=True)[39]
                
            aa_score_0_1 = evaluate_with_threshold(aa_score, best_threshold)
            aa_score_list.append(aa_score)
            aa_score_0_1_list.append(aa_score_0_1)    
            color_pse(pdb_file, pre_ply_path, aa_score, best_threshold, out_name)
            df = pd.DataFrame({'aa_id':range(1, len(aa_score)+1),'score':aa_score})
            df.to_csv(os.path.join(pre_ply_path, out_name+'.csv'),sep=',',index=False)
        

