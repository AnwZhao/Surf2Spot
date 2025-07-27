from masif.my_generate_ply import compute_inp_surface
import os
import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
# python -i /home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_Z_most_chain -o /home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_Z_most_chain

os.environ["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masif/extra_dependence/APBS-3.4.1/lib')
msms_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masif/extra_dependence/msms/msms.x86_64Linux2.2.6.1')
apbs_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masif/extra_dependence/APBS-3.4.1/bin/apbs')
pdb2pqr_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masif/extra_dependence/pdb2pqr/pdb2pqr")
multivalue_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "masif/extra_dependence/APBS-3.4.1/share/apbs/tools/bin/multivalue")

############ Set to your own path! ############
#outdir = '/home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_Z_most_chain'
#indir = '/home/anwzhao/my_development/HSFilter/GNN-M_data/all_data/preprocess_pdb_chain_Z_most_chain'
########################

def process_one_pdb(pdb, input_dir, output_dir, probe):
    try:
        prot_chain = pdb.split('_')[1].split('.')[0]
        pdb_name = pdb.split('_')[0]
        pdb_path = os.path.join(input_dir, f'{pdb_name}_{prot_chain}.pdb')
        compute_inp_surface(pdb_path, msms_bin, apbs_bin, pdb2pqr_bin, multivalue_bin, prot_chain, output_dir, probe)
    except Exception as e:
        print(f"Failed to process {pdb}: {e}")
        

def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o','--output_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    parser.add_argument('--probe', type=str, default = '5.0')
    args = parser.parse_args()

    # 获取待处理的PDB文件
    inpdb_list = sorted([
        pdb for pdb in os.listdir(args.input_dir)
        if pdb.endswith('.pdb') and not os.path.exists(os.path.join(args.input_dir, pdb.split('.pdb')[0] + '_all_5.0.ply'))
    ])

    print(f"Found {len(inpdb_list)} PDBs to process.")

    # 并发处理，每轮最多10个
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(process_one_pdb, pdb, args.input_dir, args.output_dir, args.probe)
            for pdb in inpdb_list
        ]
        for future in as_completed(futures):
            future.result()  # 获取异常（如果有）

main()
