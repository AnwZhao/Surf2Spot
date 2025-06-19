import os
import argparse
import pandas as pd
import linecache
import re
from transformers import T5Tokenizer, T5EncoderModel
from Surf2Spot.data.predict_from_pdb import run_gpsite
import shutil
import glob
#python ./script/predict_from_pdb.py -i ./example/pdb_input_test/3vjj_single.pdb -o ./example/pdb_input_test/ --gpu 0

def sort_score(txt_path):
    contents = linecache.getlines(txt_path)
    #print(contents)
    No=[]
    AA=[]
    DNA_binding=[]
    RNA_binding=[]
    Peptide_binding=[]
    Protein_binding=[]
    ATP_binding=[]
    HEM_binding=[]
    ZN_binding=[]
    CA_binding=[]
    MG_binding=[]
    MN_binding=[]
    for i in range(len(contents)):
        c=contents[i][:-1]
        if i == 0:
            col_name=c.split('\t')
            #print(col_name)
        else:
            line_list=c.split('\t')
            No.append(line_list[0])
            AA.append(line_list[1])
            DNA_binding.append(line_list[2])
            RNA_binding.append(line_list[3])
            Peptide_binding.append(line_list[4])
            Protein_binding.append(line_list[5])
            ATP_binding.append(line_list[6])
            HEM_binding.append(line_list[7])
            ZN_binding.append(line_list[8])
            CA_binding.append(line_list[9])
            MG_binding.append(line_list[10])
            MN_binding.append(line_list[11])

    df = pd.DataFrame({'No':No,'AA':AA,'DNA_binding':DNA_binding,'RNA_binding':RNA_binding,'Peptide_binding':Peptide_binding,
				'Protein_binding':Protein_binding,'ATP_binding':ATP_binding,'HEM_binding':HEM_binding,'ZN_binding':ZN_binding,
				'CA_binding':CA_binding,'MG_binding':MG_binding,'MN_binding':MN_binding})
    df = df.sort_values(by = ['Protein_binding','Peptide_binding'],ascending=False)
    df['No'] = df['No'].astype(int)
    #out_csv_path = txt_path.split('/')[-1][:-4]+'.csv'
    #df.to_csv(os.path.join(out_path,out_csv_path),index=False)

    #df[((df['No']>=163) & (df['No']<=172)) | (((df['No']>=186) & (df['No']<=246))) ]

    return df


def get_dssp_surface(dssp_line_list):
    i=0
    res_num_list = []
    chain_id_list = []
    ACC_list = []
    for line in dssp_line_list:
        if i == 1 and line != '\n' and line != '':
            res_num_list.append(line[5:10].strip())
            chain_id_list.append(line[10:13].strip())
            ACC_list.append(line[34:38].strip())
        if line.strip()[0] == '#':
            i = 1

    t_r = []
    t_c = []
    t_a = []
    for i in range(len(res_num_list)):
        if res_num_list[i]!='':
            t_r.append(res_num_list[i])
            t_c.append(chain_id_list[i])
            t_a.append(ACC_list[i])
    res_num_list = t_r
    chain_id_list = t_c
    ACC_list = t_a
    
    df = pd.DataFrame({'res_num':res_num_list,'chain_id':chain_id_list,'ACC':ACC_list})
    df['ACC'] = df['ACC'].astype(int)
    df['res_num'] = df['res_num'].astype(int)
    df = df[df['ACC']>=15]
    return df

def clean_empty_list(list):
    clean_data = [x for x in list if x is not None and x != '']
    return clean_data

def get_dssp_surface_in_contigs(dssp_line_list,contigs):
    i=0
    res_num_list = []
    chain_id_list = []
    ACC_list = []
    for line in dssp_line_list:
        if i == 1 and line != '\n' and line != '':
            if len(clean_empty_list(line[5:10].split(' '))) == 0:
                pass
            else:
                res_num_list.append(line[5:10].strip())
                chain_id_list.append(line[10:13].strip())
                ACC_list.append(line[34:38].strip())
        if clean_empty_list(line.split(' '))[0] == '#':
            i = 1
    df = pd.DataFrame({'res_num':res_num_list,'chain_id':chain_id_list,'ACC':ACC_list})
    df['ACC'] = df['ACC'].astype(int)
    df['res_num'] = df['res_num'].astype(int)
    df = df[df['ACC']>=15]

    t = 0
    for range_list in contigs:
        df_temp = df[(df['res_num'] >= range_list[0]) & (df['res_num'] <= range_list[1])]
        if t == 0:
            df_concat = df_temp
            t = 1
        else:
            df_concat = pd.concat([df_concat,df_temp])

    return df_concat

def get_contigs_list(contigs):
    contigs_motif = contigs.split('/')  #args.contigs ='A1-194/A501-694/0'
    res_range_list = []
    for i in range(len(contigs_motif) - 1):
        chain_id = ''.join(re.findall(r'[A-Za-z]', contigs_motif[i]))
        res_range = re.findall("\d+\.?\d*", contigs_motif[i])
        res_range = list(map(int, res_range))
        res_range_list.append(res_range)
    return res_range_list  #[[start1,end1],[start2,end2]...]

'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i', '--pdb_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o', '--out_dir', type=str, metavar='',
                        help='Directory of output files.(Please enter an absolute path)')
    parser.add_argument('--gpu', type = str, default = '0',
                        help='which gpu to use')
    args = parser.parse_args()
    #args = parser.parse_args(args=[])

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    ############ Set to your own path! ############
    ProtTrans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_emb/prot_t5_xl_half_uniref50-enc")
    ###############################################
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
    print('model loaded')
    
    file_list = sorted(os.listdir(args.pdb_dir))
    surface_HS_path = os.path.join(args.out_dir,'surface_HS')
    if not os.path.exists(surface_HS_path):
        os.mkdir(surface_HS_path)

    dssp_path = os.path.join(args.out_dir,'dssp')
    if not os.path.exists(dssp_path):
        os.mkdir(dssp_path)

    for f in file_list:
        if f.endswith('.pdb') and f.split('.')[0] not in os.listdir(surface_HS_path):
            print(f)
            old_file_list = os.listdir(args.out_dir)
            run_gpsite(os.path.join(args.pdb_dir,f), args.out_dir, args.gpu, model, tokenizer)
            #os.system(f'python /data/zaw/development/HSfilter/get_GPSite_score/2.predict_from_pdb.py -i {os.path.join(args.pdb_dir,f)} -o {args.out_dir} --gpu {args.gpu}')
            new_file_list = os.listdir(args.out_dir)

            # 使用集合求差集
            differ = set(new_file_list) - set(old_file_list)

            # 将差集转换回列表
            differ = list(differ)
            pdb_p = 'pdb'
            ppi_p = 'pred'
            dssp_p = 'DSSP'
            for d in differ:
                if os.path.isdir(os.path.join(args.out_dir,d)):
                    out_dssp_name = f[:-4]+'.dssp'
                    ###os.system(f'mkdssp -i {os.path.join(args.out_dir,d,pdb_p,f)} -o {os.path.join(args.out_dir,d,dssp_p,out_dssp_name)}')
                    #dssp-->~/my_development/RFtools/GPSite/example/pdb_input_test/2024-05-06_15-36-55/DSSP/*.dssp

                    dssp_line_list = linecache.getlines(os.path.join(args.out_dir,d,dssp_p,out_dssp_name))
                    df_dssp_more_15_in_tareget_site = get_dssp_surface(dssp_line_list)
                    df_ppi = sort_score(os.path.join(args.out_dir,d,ppi_p,f[:-4]+'.txt'))
                    #ppi_table-->/home/anwzhao/my_development/RFtools/GPSite/example/pdb_input_test/2024-05-06_15-36-55/pred/3vjj_A.txt

                    #print(df_ppi)
                    #print(df_dssp_more_15_in_tareget_site['res_num'])
                    df = df_ppi[df_ppi["No"].isin(list(df_dssp_more_15_in_tareget_site['res_num']))]
                    #print(df)
                    df['Protein_binding'] = df['Protein_binding'].astype(float)
                    df_Protein = df.sort_values(by='No', ascending=True)
                    out_f = os.path.join(surface_HS_path,f[:-4])
                    os.mkdir(out_f)
                    shutil.copy(os.path.join(args.out_dir, d, pdb_p, f), out_f)
                    df_Protein.to_csv(os.path.join(out_f, 'GPSite_score.csv'), index = False)
        elif f.endswith('.pdb'):
            print(f'-----{f} exists-----')
    # 源目录和目标目录
    source_pattern = os.path.join(args.out_dir,'20*/DSSP/*.dssp')  # 使用通配符匹配
    destination_dir = os.path.join(args.out_dir,'dssp')

    # 获取符合条件的文件列表
    files = glob.glob(source_pattern)

    # 确保目标目录存在
    os.makedirs(destination_dir, exist_ok=True)

    # 复制文件
    for file in files:
        shutil.copy(file, destination_dir)

main()
'''

def get_gpsite_split(pdb_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ############ Set to your own path! ############
    ProtTrans_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_emb/prot_t5_xl_half_uniref50-enc")
    ###############################################
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
    print('model loaded')
    
    file_list = sorted(os.listdir(pdb_dir))
    surface_HS_path = os.path.join(out_dir,'surface_HS')
    if not os.path.exists(surface_HS_path):
        os.mkdir(surface_HS_path)

    dssp_path = os.path.join(out_dir,'dssp')
    if not os.path.exists(dssp_path):
        os.mkdir(dssp_path)

    for f in file_list:
        if f.endswith('.pdb') and f.split('.')[0] not in os.listdir(surface_HS_path):
            print(f)
            old_file_list = os.listdir(out_dir)
            run_gpsite(os.path.join(pdb_dir,f), out_dir, '0', model, tokenizer)
            new_file_list = os.listdir(out_dir)

            # 使用集合求差集
            differ = set(new_file_list) - set(old_file_list)

            # 将差集转换回列表
            differ = list(differ)
            pdb_p = 'pdb'
            ppi_p = 'pred'
            dssp_p = 'DSSP'
            for d in differ:
                if os.path.isdir(os.path.join(out_dir,d)):
                    out_dssp_name = f[:-4]+'.dssp'
                    ###os.system(f'mkdssp -i {os.path.join(args.out_dir,d,pdb_p,f)} -o {os.path.join(args.out_dir,d,dssp_p,out_dssp_name)}')
                    #dssp-->~/my_development/RFtools/GPSite/example/pdb_input_test/2024-05-06_15-36-55/DSSP/*.dssp

                    dssp_line_list = linecache.getlines(os.path.join(out_dir,d,dssp_p,out_dssp_name))
                    df_dssp_more_15_in_tareget_site = get_dssp_surface(dssp_line_list)
                    df_ppi = sort_score(os.path.join(out_dir,d,ppi_p,f[:-4]+'.txt'))
                    #ppi_table-->/home/anwzhao/my_development/RFtools/GPSite/example/pdb_input_test/2024-05-06_15-36-55/pred/3vjj_A.txt

                    #print(df_ppi)
                    #print(df_dssp_more_15_in_tareget_site['res_num'])
                    df = df_ppi[df_ppi["No"].isin(list(df_dssp_more_15_in_tareget_site['res_num']))]
                    #print(df)
                    df['Protein_binding'] = df['Protein_binding'].astype(float)
                    df_Protein = df.sort_values(by='No', ascending=True)
                    out_f = os.path.join(surface_HS_path,f[:-4])
                    os.mkdir(out_f)
                    shutil.copy(os.path.join(out_dir, d, pdb_p, f), out_f)
                    df_Protein.to_csv(os.path.join(out_f, 'GPSite_score.csv'), index = False)
        elif f.endswith('.pdb'):
            print(f'-----{f} exists-----')
    # 源目录和目标目录
    source_pattern = os.path.join(out_dir,'20*/DSSP/*.dssp')  # 使用通配符匹配
    destination_dir = os.path.join(out_dir,'dssp')

    # 获取符合条件的文件列表
    files = glob.glob(source_pattern)

    # 确保目标目录存在
    os.makedirs(destination_dir, exist_ok=True)

    # 复制文件
    for file in files:
        shutil.copy(file, destination_dir)