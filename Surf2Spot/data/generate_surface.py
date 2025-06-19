from masif.my_generate_ply import compute_inp_surface
import os
import argparse
import shutil
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

def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o','--output_dir', type=str, metavar='', required= True,
                        help='Directory of output files.(Please enter an absolute path)')
    parser.add_argument('--probe', type=str, default = '5.0')
    args = parser.parse_args()

    inpdb_list = sorted(os.listdir(args.input_dir))
    for pdb in inpdb_list:
        if pdb.endswith('.pdb') and os.path.exists(os.path.join(args.input_dir, pdb.split('.pdb')[0]+'_all_5.0.ply')) == False :  #1a22_B.pdb
            print(pdb)
            prot_path = os.path.join(args.input_dir,pdb)
            prot_chain = pdb.split('_')[1].split('.')[0]
            print(prot_chain)        
            pdb_name = pdb.split('_')[0]
            try:
                compute_inp_surface(os.path.join(args.input_dir,f'{pdb_name}_{prot_chain}.pdb'), msms_bin,apbs_bin,pdb2pqr_bin,multivalue_bin, prot_chain, args.output_dir, args.probe)
            except:
                pass

main()