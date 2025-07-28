import torch
import esm
import os
from Bio import SeqIO  # 用于读取 fasta 文件
import argparse  
   

# 取消设置LD_LIBRARY_PATH环境变量
os.environ['LD_LIBRARY_PATH'] = ''  # unset LD_LIBRARY_PATH
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:15004'  # 示例值，你可以根据需要修改
 
# 验证设置是否成功
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

# 加载模型
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# 读取 fasta 文件并提取所有序列
def read_fasta(fasta_file):
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)  # 保存序列id和对应的序列
    return sequences

# 对每个序列进行预测并保存结果
def predict_and_save(fasta_file, output_dir):
    sequences = read_fasta(fasta_file)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 批量处理序列
    for seq_id, sequence in sequences.items():
        print(f"Processing sequence: {seq_id}")
        
        with torch.no_grad():
            output = model.infer_pdb(sequence)

        # 保存 PDB 结果文件
        if seq_id.endswith('.pdb'):
            seq_id = seq_id[:-4]
        if seq_id.startswith('>'):
            seq_id = seq_id[1:]
        output_file = os.path.join(output_dir, f"{seq_id}.pdb")
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Saved PDB for {seq_id} to {output_file}")


parser = argparse.ArgumentParser()

parser.add_argument("-i", '--input')
parser.add_argument("-o", '--output')


args = parser.parse_args()
# 使用示例
fasta_file = args.input  # 你的 fasta 文件路径
output_dir = args.output  # 预测结果保存的输出目录

# 调用函数进行批量预测
predict_and_save(fasta_file, output_dir)
