import os
import typer
import subprocess
from pathlib import Path
from Surf2Spot.tools.path_check import path_exists, mk_dir
from Surf2Spot.data.get_domain import get_domain_tsv
from Surf2Spot.data.preprocess_single_chain import get_single_chain
from Surf2Spot.data.get_atom_feature import atom_feature_engineering
from Surf2Spot.data.extract_sequences import extract_sequences_from_pdb, process_esmpdb
from Surf2Spot.data.get_prottrans_embed import run_prottrans
from Surf2Spot.data.get_surface_partzone_antighs import NB_surfpart
from Surf2Spot.predict.get_data_list_antighs import NB_predict
from Surf2Spot.draw.draw_predict_antighs import NB_draw

from Surf2Spot.data.batch_predict_from_pdb_and_dssp_surface import get_gpsite_split
from Surf2Spot.data.get_surface_partzone_hsfilter import HS_surfpart
from Surf2Spot.predict.get_data_list_hsfilter import HS_predict
from Surf2Spot.draw.draw_predict_hsfilter import HS_draw


def run_NB_preprocess(
    input_dir: Path = typer.Option(..., "-i", "--input", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", "--output", help="Output directory of preprocessed files."),
    esm_fasta: Path = typer.Option(None,"--esm", help="Set the fasta file path of the input sequence if esmfold prediction is needed."),
    domain_out_tsv: Path = typer.Option(None,'-ds',"--domain_split", help="The output path of domain segmentation result(.tsv)."),
    min_domain_length: int = typer.Option(30, "--min_domain_length", help="Minimum length threshold of domain segmentation, which defaults to 30.")
):
    
    print('running NB-preprocess')
    
    if domain_out_tsv is None:
        domain_out_tsv = os.path.join(os.path.dirname(input_dir), 'chainsaw.tsv')
        
    if esm_fasta is None:
        if path_exists(input_dir):
            get_single_chain(input_dir, output_dir)
            get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)

    else:
        if path_exists(esm_fasta):
            print('===================running esmfold===================')
            esm_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools/predict_esm.py')
            subprocess.run(f"conda run -n surf2spot_tools python {esm_code} -i {esm_fasta} -o {input_dir} ", shell=True)
            process_esmpdb(input_dir)
            get_single_chain(input_dir, output_dir)
            get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)
            

def run_NB_craft(
    output_dir: Path = typer.Option(..., "-i", help="Output directory."),
    seq_fasta: Path = typer.Option(None,"-s", help="Outputs the path of the sequence result as the input of prottrans."),
    domain_out_tsv: Path = typer.Option(None, "-ds", "--domain_split", help="The input path of domain segmentation result(.tsv)."),
    emb_out_path: Path = typer.Option(None, "-emb", help="The output path of ProtTrans embedding."),
    split_domain_length: int = typer.Option(400,"--split_domain_length", help="The length threshold of the protein for domain segmentation or not, which defaults to 400."),
    probe: float = typer.Option(5.0, "--probe", help="Set the probe radius, which is 5.0 by default.")
):
    
    print('running NB-craft')
    
    if domain_out_tsv is None:
        domain_out_tsv = os.path.join(os.path.dirname(output_dir), 'chainsaw.tsv')
    if emb_out_path is None:
        emb_out_path = os.path.join(os.path.dirname(output_dir), 'seq_prottrans.h5')
    if seq_fasta is None:
        seq_fasta = os.path.join(os.path.dirname(output_dir), 'seq.fasta')
    
    if path_exists(domain_out_tsv) and path_exists(output_dir):
        extract_sequences_from_pdb(output_dir, seq_fasta)
        run_prottrans(seq_fasta, emb_out_path)
        surface_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/generate_surface.py')
        print('===================running surface calculation===================')
        check_surf2spot_tools_dependency()
        subprocess.run(f"conda run -n surf2spot_tools python {surface_code} -i {output_dir} -o {output_dir} --probe {probe}", shell=True)
        atom_feature_engineering(output_dir)
        NB_surfpart(output_dir, domain_out_tsv, split_domain_length)


def run_NB_predict(
    input_dir: Path = typer.Option(..., "-i", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", help="Output directory."),
    emb_path: Path = typer.Option(None, "-emb", "--emb_path", help="The output path of ProtTrans embedding."),
    model_path: Path = typer.Option(...,"--model", help="The model weight path."),
    threshold: float = typer.Option(0.4,"--threshold", help="Segmentation threshold.") 
):
    print('running NB-predict')

    if emb_path is None:
        emb_path = os.path.join(os.path.dirname(input_dir), 'seq_prottrans.h5')
        
    if path_exists(emb_path) and path_exists(input_dir) and path_exists(model_path):
        mk_dir(output_dir)
        NB_predict(input_dir, output_dir, emb_path, model_path, threshold)


def run_NB_draw(
    input_dir: Path = typer.Option(..., "-i", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", help="Output directory."),
):
    print('running NB-draw')
    if path_exists(input_dir) and path_exists(output_dir):
        NB_draw(input_dir, output_dir)


###################################################################################


def run_HS_preprocess(
    input_dir: Path = typer.Option(..., "-i", "--input", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", "--output", help="Output directory."),
    esm_fasta: Path = typer.Option(None, "--esm", help="Set the fasta file path of the input sequence if esmfold prediction is needed."),
    domain_out_dir: Path = typer.Option(None, "-ds2", help="The output path of domain segmentation result(directory, GPSite)."),
    domain_out_tsv: Path = typer.Option(None, '-ds1', help="The output path of domain segmentation result(.tsv, Chainsaw)."),
    min_domain_length: int = typer.Option(30, "--min_domain_length", help="Minimum length threshold of domain segmentation, which defaults to 30."),
    split_type: int = typer.Option(1, "--split", help="Set the segmentation method: 1 --> Chainsaw ; 2 --> GPSite.")
    
):
    print('running HS-preprocess')
    if split_type == 1:
        if domain_out_tsv is None:
            domain_out_tsv = os.path.join(os.path.dirname(input_dir), 'chainsaw.tsv')
    elif split_type == 2:
        if domain_out_dir is None:
            domain_out_dir = os.path.join(os.path.dirname(input_dir), 'GPscore')
    else:
        raise ValueError("The segmentation method is defined incorrectly! Set the segmentation method: 1 --> chainsaw; 2 --> gpsite")

    if split_type == 2:
        if esm_fasta is None:
            if path_exists(input_dir):
                get_single_chain(input_dir, output_dir)
                get_gpsite_split(output_dir, domain_out_dir)
        else:
            if path_exists(esm_fasta):
                print('===================running esmfold===================')
                esm_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools/predict_esm.py')
                subprocess.run(f"conda run -n surf2spot_tools python {esm_code} -i {esm_fasta} -o {input_dir} ", shell=True)
                process_esmpdb(input_dir)
                get_single_chain(input_dir, output_dir)
                get_gpsite_split(output_dir, domain_out_dir)


    elif split_type == 1:
        if esm_fasta is None:
            if path_exists(input_dir):
                get_single_chain(input_dir, output_dir)
                get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)
        else:
            if path_exists(esm_fasta):
                print('===================running esmfold===================')
                esm_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools/predict_esm.py')
                subprocess.run(f"conda run -n surf2spot_tools python {esm_code} -i {esm_fasta} -o {input_dir} ", shell=True)
                get_single_chain(input_dir, output_dir)
                get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)



            
            

def run_HS_craft(
    output_dir: Path = typer.Option(..., "-i", help="Output directory."),
    domain_out_dir: Path = typer.Option(None, "-ds2", help="The output directory of gpsite amino acid score."),
    seq_fasta: Path = typer.Option(None, "-s", help="Outputs the path of the sequence result as the input of prottrans."),
    emb_out_path: Path = typer.Option(None, "-emb", "--emb_path", help="Minimum length threshold of domain segmentation, which defaults to 30."),
    split_top_rate: float = typer.Option(0.66, "--split_top_rate", help="Select the top percentage for further screening(GPSite)."),
    domain_out_tsv: Path = typer.Option(None, "-ds1", help="The input path of domain segmentation result(.tsv)."),
    split_domain_length: int = typer.Option(400,"--split_domain_length", help="The length threshold of the protein for domain segmentation or not, which defaults to 400."),
    probe: float = typer.Option(5.0, "--probe", help="Set the probe radius, which is 5.0 by default."),
    split_type: int = typer.Option(1, "--split", help="Set the segmentation method: 1 --> Chainsaw ; 2 --> GPSite.")
):
    print('running HS-craft')

    if emb_out_path is None:
        emb_out_path = os.path.join(os.path.dirname(output_dir), 'seq_prottrans.h5')
    if seq_fasta is None:
        seq_fasta = os.path.join(os.path.dirname(output_dir), 'seq.fasta')

    if split_type == 1:
        if domain_out_tsv is None:
            domain_out_tsv = os.path.join(os.path.dirname(output_dir), 'chainsaw.tsv')
    elif split_type == 2:
        if domain_out_dir is None:
            domain_out_dir = os.path.join(os.path.dirname(output_dir), 'GPscore')
    else:
        raise ValueError("The segmentation method is defined incorrectly! Set the segmentation method: 1 --> chainsaw; 2 --> gpsite")

    if split_type == 2:
        if path_exists(domain_out_dir) and path_exists(output_dir):
            extract_sequences_from_pdb(output_dir, seq_fasta)
            run_prottrans(seq_fasta, emb_out_path)
            surface_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/generate_surface.py')
            print('===================running surface calculation===================')
            check_surf2spot_tools_dependency()
            subprocess.run(f"conda run -n surf2spot_tools python {surface_code} -i {output_dir} -o {output_dir} --probe {probe}", shell=True)
            atom_feature_engineering(output_dir)
            HS_surfpart(output_dir, domain_out_dir, split_top_rate)

    elif split_type == 1:    
        if path_exists(domain_out_tsv) and path_exists(output_dir):
            extract_sequences_from_pdb(output_dir, seq_fasta)
            run_prottrans(seq_fasta, emb_out_path)
            surface_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/generate_surface.py')
            print('===================running surface calculation===================')
            check_surf2spot_tools_dependency()
            subprocess.run(f"conda run -n surf2spot_tools python {surface_code} -i {output_dir} -o {output_dir} --probe {probe}", shell=True)
            atom_feature_engineering(output_dir)
            NB_surfpart(output_dir, domain_out_tsv, split_domain_length)


        


def run_HS_predict(
    input_dir: Path = typer.Option(..., "-i", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", help="Output directory."),
    emb_path: Path = typer.Option(None, "-emb", "--emb_path", help=" output path of ProtTrans embedding."),
    model_path: Path = typer.Option(..., "--model", help="The model weight path."),
    threshold: float = typer.Option(0.4,"--threshold", help="Segmentation threshold.") 
):
    print('running HS-predict')

    if emb_path is None:
        emb_path = os.path.join(os.path.dirname(input_dir), 'seq_prottrans.h5')
    
    if path_exists(emb_path) and path_exists(input_dir) and path_exists(model_path):
        mk_dir(output_dir)
        HS_predict(input_dir, output_dir, emb_path, model_path, threshold)




def run_HS_draw(
    input_dir: Path = typer.Option(..., "-i", help="Input directory of pdb file."),
    output_dir: Path = typer.Option(..., "-o", help="Output directory."),
):
    print('running HS-draw')
    if path_exists(input_dir) and path_exists(output_dir):
        HS_draw(input_dir, output_dir)


def check_surf2spot_tools_dependency():
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/masif/extra_dependence')):
        print(f"❌ '{dir_path}' Directory does not exist. Please configure MSMS(2.6.1), APBS(3.4.1) and PDB2PQR(2.1.1) to this directory correctly.")
        sys.exit(1)  
