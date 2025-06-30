import numpy as np
import torch
import json
import os
import re
import h5py
import pandas as pd
from biopandas.pdb import PandasPdb
from Bio.PDB import PDBParser
import random
import argparse
from Bio import SeqIO
from sklearn.metrics import pairwise_distances
from h5py import Dataset, Group, File
from torch_geometric.nn import DynamicEdgeConv
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial import KDTree,distance
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
import warnings
# 忽略所有的 DeprecationWarning 警告
warnings.filterwarnings('ignore')

def get_ca_coords(pdb='1LUC', chain='A'):
    df = PandasPdb().read_pdb(pdb).df['ATOM']
    df = df[df['atom_name']== 'CA']
    return df

def luciferase_contact_map(pdb, chain, seq_gap=4, contact_cutoff=6):
    # download pdb and save ca coordinates
    ca_coords = get_ca_coords(pdb, chain)
    # pairwise distances
    dist_arr = pairwise_distances(ca_coords[['x_coord', 'y_coord', 'z_coord']].values)
    # remove neighboring residues 
    
    # 找到最大列数
    max_cols = 1024 # 这里是单一矩阵
    # 扩充矩阵（假设是对角矩阵，填充大值）
    max_num = np.max(dist_arr)*2
    if dist_arr.shape[1] < max_cols:
        padded_matrix = np.pad(dist_arr, 
                           ((0, 0), (0, max_cols - dist_arr.shape[1])), 
                           mode='constant', 
                           constant_values=max_num)
    else:
        padded_matrix = dist_arr
        
    min_val = np.min(padded_matrix)
    max_val = np.max(padded_matrix)
    normalized_matrix = (padded_matrix - min_val) / (max_val - min_val)
    return normalized_matrix

# 解析点云数据和氨基酸数据
def parse_ply(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    vertex_data = []
    face_data = []
    reading_vertices = False
    reading_faces = False
    
    for line in lines:
        if line.startswith("end_header"):
            reading_vertices = True
            continue
        elif reading_vertices:
            parts = line.strip().split()
            if len(parts) == 10:  # vertex data
                vertex_data.append([float(p) for p in parts])
            elif len(parts) != 10:  # face data starts
                reading_vertices = False
                reading_faces = True
        
        if reading_faces:
            parts = line.strip().split()
            face_data.append([int(p) for p in parts[1:]])  # ignore the first element (number of vertices in face)
    
    vertex_data = np.array(vertex_data)
    face_data = np.array(face_data)
    
    return vertex_data, face_data

def parse_amino_acid_data(file_path):
    # 解析氨基酸特性数据，假设每行包含 (x, y, z, feature1, feature2, ..., featureN)
    
    return np.genfromtxt(file_path, delimiter=',')


def count_atoms_per_residue(pdb_file):
    # 创建 PDB 解析器
    parser = PDBParser()
    
    # 解析 PDB 文件
    structure = parser.get_structure('protein', pdb_file)
    
    # 初始化结果列表
    atom_counts = []
    
    # 遍历模型、链和氨基酸
    for model in structure:
        for chain in model:
            for residue in chain:
                # 检查是否是氨基酸（排除水分子等）
                if residue.get_resname().strip() in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
                                                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
                                                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
                                                    'SER', 'THR', 'TRP', 'TYR', 'VAL']:
                    # 统计当前氨基酸的原子数
                    atom_count = len([atom for atom in residue])
                    atom_counts.append(atom_count)
    
    return atom_counts

def normalize_signed_columns(point_cloud_features):
    # 复制数据，避免修改原始数据
    normalized_features = point_cloud_features.copy()

    # 需要归一化的列索引（假设前 3 列是需要特殊处理的）
    cols_to_normalize = [0, 1, 2]

    for col in cols_to_normalize:
        data = point_cloud_features[:, col]

        # 处理负数部分
        neg_mask = data < 0
        if np.any(neg_mask):  # 确保存在负数
            min_neg = np.min(data[neg_mask])  # 负数中的最小值
            max_neg = np.max(data[neg_mask])  # 负数中的最大值
            if min_neg != max_neg:  # 避免除以0
                normalized_features[neg_mask, col] = (data[neg_mask] - max_neg) / (max_neg - min_neg)

        # 处理正数部分
        pos_mask = data > 0
        if np.any(pos_mask):  # 确保存在正数
            min_pos = np.min(data[pos_mask])  # 正数中的最小值
            max_pos = np.max(data[pos_mask])  # 正数中的最大值
            if min_pos != max_pos:  # 避免除以0
                normalized_features[pos_mask, col] = (data[pos_mask] - min_pos) / (max_pos - min_pos)

        # 0 值保持不变
        normalized_features[~(neg_mask | pos_mask), col] = 0

    return normalized_features

# 对齐点云数据和氨基酸特性数据
def align_data(pdb_file, point_cloud_data, amino_acid_data, progen2_data, dist_arr):
    point_cloud_coords = point_cloud_data[:, :3]
    point_cloud_features = point_cloud_data[:, 3:]
    
    amino_acid_coords = amino_acid_data[:, :3]
    amino_acid_features = amino_acid_data[:, 3:]
    
    tree = KDTree(amino_acid_coords)
    distances, indices = tree.query(point_cloud_coords)
    
    matched_amino_acid_features = amino_acid_data[indices, 3:]

    repeat_counts = count_atoms_per_residue(pdb_file)
    # 生成新 tensor 通过重复
    progen2_data = np.repeat(progen2_data, repeat_counts, axis=0)
    dist_arr = np.repeat(dist_arr, repeat_counts, axis=0)

    matched_amino_acid_features = np.hstack((matched_amino_acid_features, progen2_data[indices, :], dist_arr[indices, :]))

    point_cloud_features = normalize_signed_columns(point_cloud_features)
    combined_features = np.hstack((point_cloud_features, matched_amino_acid_features))
    
    #point_cloud_coords 中心化
    centroid = np.mean(point_cloud_coords,axis = 0)
    centralized_point_cloud_coords = point_cloud_coords-centroid
    

    
    return centralized_point_cloud_coords, combined_features


# 构建图数据
def build_graph_data(point_cloud_coords, combined_features, face_data):
    edge_index = []
    for face in face_data:
        for i in range(len(face)):
            edge_index.append([face[i], face[(i+1) % len(face)]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #print('edge_index',edge_index,edge_index.shape)
    x = torch.tensor(combined_features, dtype=torch.float)
    
    centroid = np.mean(point_cloud_coords, axis = 0)
    normalized_point = point_cloud_coords - centroid
    pos = torch.tensor(normalized_point, dtype=torch.float)
    
    return Data(x=x, pos=pos, edge_index=edge_index)


# 解析所有蛋白质数据
def parse_all_protein_data(pdb_files, protein_files, amino_acid_atom_files, emb_path):
    data_list = []
    pdb_file_list = []
    wrong_site_index = 0
    wrong_site_list = []
    protein_files_list=[]
    
    with File(emb_path,'r') as f:
    
        for pdb_file, pc_file, aa_file in zip(pdb_files, protein_files, amino_acid_atom_files):
            chain = pdb_file.split('/')[-1].split('.')[0].split('_')[1]
            dist_arr = luciferase_contact_map(pdb_file, chain)
            point_cloud_data, face_data = parse_ply(pc_file)
            amino_acid_data = parse_amino_acid_data(aa_file)[1:]
            amino_acid_data[:,4] = amino_acid_data[:,4] / 360
            amino_acid_data[:,5] = amino_acid_data[:,5] / 360
            
            amino_acid_data[:,6] = (amino_acid_data[:,6] - np.min(amino_acid_data[:,6])) / (np.max(amino_acid_data[:,6])-np.min(amino_acid_data[:,6]))  # b-factor          
            k = pc_file.split('/')[-1].split('_all_')[0]
            progen2_data = f[k][:]
       
            point_cloud_coords, combined_features = align_data(pdb_file, point_cloud_data, amino_acid_data, progen2_data, dist_arr)
            data = build_graph_data(point_cloud_coords, combined_features, face_data)
            #print(pc_file,'data_y',data.y)
            data_list.append(data)
            pdb_file_list.append(pdb_file)
            protein_files_list.append(protein_files)                     

    f.close()  
    return data_list, pdb_file_list, protein_files_list

    
class DGCNN(nn.Module):
    def __init__(self, k=20, pc_dim=42-3, output_dim=1, output_progen2_dim=16, output_dist_arr_dim=32):  #input_dim=43
        super(DGCNN, self).__init__()
        self.input_dim = pc_dim + output_dist_arr_dim + output_progen2_dim
        self.k = k
        self.conv1 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * self.input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), k=k)
        self.conv2 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ), k=k)
        self.conv3 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * 128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ), k=k)
        
        self.gat1 = GATConv(in_channels = 256+self.input_dim , out_channels = 128, heads=8, concat=True)
        self.gat2 = GATConv(in_channels = 1024 , out_channels = 64, heads=8, concat=True)
        self.gat3 = GATConv(in_channels = 512 , out_channels = 32, heads=8, concat=True)
        
        self.mlp = nn.Sequential(
            nn.Linear(512 + self.input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
        self.mlp_progen2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, output_progen2_dim)
        )
        
        self.mlp_dist_arr = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dist_arr_dim)
        )
        self.scale_factor = 5
        
        
    def forward(self, data):
        x, pos, batch, edge_index = data.x, data.pos, data.batch,data.edge_index

        x_surf = x[:,:39]
        x_progen2 = x[:,39:1063]
        x_progen2_1 = self.mlp_progen2(x_progen2) 
        x_dist_arr = x[:,1063:1063+1024]  # 1000
        x_dist_arr_1 = self.mlp_dist_arr(x_dist_arr) 
        x = torch.cat([x_surf, x_progen2_1, x_dist_arr_1], dim=1) # x_surf+pos-->42, x_prottrans_1-->16
        
        
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        x3 = torch.cat([x3, x_surf, x_progen2_1, x_dist_arr_1], dim=1)  # add_back_language_embedding_concat
        x4 = self.gat1(x3, edge_index)
        x4 = F.elu(x4) 
        x5 = self.gat2(x4, edge_index)
        x5 = F.elu(x5) 
        x6 = self.gat3(x5, edge_index)
        x6 = F.elu(x6) 
        x6 = torch.cat([x6, x3], dim=1)
        out = self.mlp(x6)  # 对每个节点的特征进行预测
        out = self.scale_factor * out
        
        out = torch.sigmoid(out)
        return out


def run_model(model, data_loader, device, threshold):
    model.eval()
    y_pred = []
    y_prob = []
    y_pred_out = []
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            #probs = torch.sigmoid(output).squeeze()
            probs = output.squeeze()
            
            preds = (probs > threshold).float()
            
            #检查转为 1 的元素个数
            num_positive = preds.sum().item()
            num = preds.shape[0]
            num_positive_threshold = 100
            
            if num*0.5 <= num_positive_threshold:
                num_positive_threshold = round(num*0.5)
            
            if num_positive < num_positive_threshold :
                # 获取前30个最大概率的索引
                top_k_indices = torch.topk(probs, num_positive_threshold).indices
                # 初始化全零的 preds
                preds = torch.zeros_like(probs)
                # 将前30个最大概率的对应位置设置为 1
                preds[top_k_indices] = 1.0

            y_pred.extend(preds.cpu().numpy())
            y_pred_out.append(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    return y_pred_out

def color_pre(in_ply,out_ply,pc_score_list):
    pc_score_list = [-i for i in pc_score_list]
    # 读取 PLY 文件
    with open(in_ply, 'r') as file:
        lines = file.readlines()

    # 查找数据段的开始和结束行
    start_idx = None
    end_idx = None
    for idx, line in enumerate(lines):
        if line.startswith('end_header'):
            start_idx = idx + 1
        if re.match(r'^\d+\s+\d+\s+\d+$', line):
            end_idx = idx
            break

    # 解析和替换 hbond 列
    for i, line in enumerate(lines[start_idx:end_idx]):
        columns = line.split()
        if len(columns) >= 5:
            columns[4] = str(pc_score_list[i])
        lines[start_idx + i] = ' '.join(columns) + '\n'

    # 写入新的 PLY 文件
    with open(out_ply, 'w') as file:
        file.writelines(lines)
    print('done')

def draw_ply(data_path, test_pdb_name_list, y_pred, out_path):
    for i in range(len(y_pred)):
        test_ply = test_pdb_name_list[i].split('/')[-1]    # 1kxq_A_all_5.0_filtered_domain_1.ply  1mel_L_all_5.0_filtered.ply
        test_ply_path = os.path.join(data_path, test_ply)   
        if '_domain_' in test_ply_path:
            out_ply_path = os.path.join(out_path, test_ply.split('_all_')[0] + '_domain_' + test_ply.split('_domain_')[-1].split('.')[0] + '_pred.ply')  ## 1kxq_A_domain_1_pred.ply
        else:
            out_ply_path = os.path.join(out_path, test_ply.split('_all_')[0] + '_pred.ply')
        color_pre(test_ply_path, out_ply_path, y_pred[i].tolist())



def NB_predict(input_dir, output_dir, emb, model, threshold):   
    data_path = input_dir
    emb_path = emb
    threshold = threshold
    model_path = model
    predict_path = output_dir

    protein_files = []
    pdb_files = []
    amino_acid_atom_files = []
    pdb_name_list = []

    data_f = os.listdir(data_path)

    for file in data_f:
        if file.endswith('.ply') and '_all_5.0_filtered' in file:  #1a22_A.pdb
            pdb_name = file.split('_all_5.0')[0]
            if os.path.exists(os.path.join(data_path,pdb_name + '_all_5.0_filtered.ply')):
                pdb_name_list.append(pdb_name)
                pdb_files.append(os.path.join(data_path, pdb_name + '.pdb'))
                protein_files.append(os.path.join(data_path, pdb_name + '_all_5.0_filtered.ply'))
                amino_acid_atom_files.append(os.path.join(data_path,pdb_name + '_combined_info_onehot_atom.csv'))
            else:
                for i in range(20):# 1kxq_A_all_5.0_filtered_domain_1.ply
                    if os.path.exists(os.path.join(data_path,pdb_name + f'_all_5.0_filtered_domain_{i}.ply')):
                        pdb_name_list.append(pdb_name)
                        pdb_files.append(os.path.join(data_path, pdb_name + '.pdb'))
                        protein_files.append(os.path.join(data_path, pdb_name + f'_all_5.0_filtered_domain_{i}.ply'))
                        amino_acid_atom_files.append(os.path.join(data_path,pdb_name + '_combined_info_onehot_atom.csv'))

    data_list, pdb_file_list, protein_files_list = parse_all_protein_data(pdb_files, protein_files, amino_acid_atom_files, emb_path)
    print('--------------------------')

    data_loader = DataLoader(data_list, batch_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # 加载模型
    model = DGCNN()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    print('running model')
    y_pred = run_model(model, data_loader, device, threshold)
    print('finish model')
    draw_ply(data_path, protein_files, y_pred, predict_path)
    print('done.')


