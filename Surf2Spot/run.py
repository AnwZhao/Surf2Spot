import os
import typer
import subprocess
from pathlib import Path
from Surf2Spot.tools.path_check import path_exists, mk_dir
from Surf2Spot.data.get_domain import get_domain_tsv
from Surf2Spot.data.preprocess_single_chain import get_single_chain
from Surf2Spot.data.get_atom_feature import atom_feature_engineering
from Surf2Spot.data.extract_sequences import extract_sequences_from_pdb
from Surf2Spot.data.get_prottrans_embed import run_prottrans
from Surf2Spot.data.get_surface_partzone_antighs import NB_surfpart
from Surf2Spot.predict.get_data_list_antighs import NB_predict
from Surf2Spot.draw.draw_predict_antighs import NB_draw

from Surf2Spot.data.batch_predict_from_pdb_and_dssp_surface import get_gpsite_split
from Surf2Spot.data.get_surface_partzone_hsfilter import HS_surfpart
from Surf2Spot.predict.get_data_list_hsfilter import HS_predict
from Surf2Spot.draw.draw_predict_hsfilter import HS_draw


def run_NB_preprocess(
    input_dir: Path = typer.Option(..., "-i", "--input", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", "--output", help="输出目录"),
    esm_fasta: Path = typer.Option(None,"--esm", help="如需要esmfold预测，输入序列的fasta文件路径"),
    domain_out_tsv: Path = typer.Option(...,'-ds',"--domain_split", help="输出的domain分割tsv"),
    min_domain_length: int = typer.Option(30, "--min_domain_length", help="domain分割的最小长度阈值，默认为 30")
):
    print('running NB-preprocess')
    if esm_fasta is None:
        if path_exists(input_dir):
            get_single_chain(input_dir, output_dir)
            get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)

    else:
        if path_exists(esm_fasta):
            print('===================running esmfold===================')
            subprocess.run(f"conda run -n surf2spot_tools esm-fold -i {esm_fasta} -o {input_dir} --cpu-only", shell=True)
            get_single_chain(input_dir, output_dir)
            get_domain_tsv(output_dir, domain_out_tsv, min_domain_length)
            

def run_NB_craft(
    output_dir: Path = typer.Option(..., "-i", help="输出目录"),
    seq_fasta: Path = typer.Option(...,"-s", help="输出的序列路径，作为prottrans的输入"),
    domain_out_tsv: Path = typer.Option(..., "-ds", "--domain_split", help="输入的domain分割tsv"),
    emb_out_path: Path = typer.Option(..., "-emb", help="prottrans输出路径"),
    split_domain_length: int = typer.Option(400,"--split_domain_length", help="进行domain分割的最小蛋白长度阈值，默认为 400"),
    probe: float = typer.Option(5.0, "--probe", help="设置探针半径，默认为 5.0")
):
    print('running NB-craft')
    if path_exists(domain_out_tsv) and path_exists(output_dir):
        extract_sequences_from_pdb(output_dir, seq_fasta)
        run_prottrans(seq_fasta, emb_out_path)
        surface_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/generate_surface.py')
        print('===================running surface calculation===================')
        subprocess.run(f"conda run -n surf2spot_tools python {surface_code} -i {output_dir} -o {output_dir} --probe {probe}", shell=True)
        atom_feature_engineering(output_dir)
        NB_surfpart(output_dir, domain_out_tsv, split_domain_length)


def run_NB_predict(
    input_dir: Path = typer.Option(..., "-i", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", help="输出目录"),
    emb_path: Path = typer.Option(..., "-emb", "--emb_path", help="prottrans的输出"),
    model_path: Path = typer.Option(...,"--model", help="模型权重"),
    threshold: float = typer.Option(0.4,"--threshold", help="分割阈值") 
):
    print('running NB-predict')
    if path_exists(emb_path) and path_exists(input_dir) and path_exists(model_path):
        mk_dir(output_dir)
        NB_predict(input_dir, output_dir, emb_path, model_path, threshold)


def run_NB_draw(
    input_dir: Path = typer.Option(..., "-i", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", help="输出目录"),
):
    print('running NB-draw')
    if path_exists(input_dir) and path_exists(output_dir):
        NB_draw(input_dir, output_dir)


###################################################################################


def run_HS_preprocess(
    input_dir: Path = typer.Option(..., "-i", "--input", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", "--output", help="输出目录"),
    esm_fasta: Path = typer.Option(None, "--esm", help="如需要esmfold预测，输入序列的fasta文件路径"),
    domain_out_dir: Path = typer.Option(..., "-gpsite_dir", help="输出的domain分割tsv")
):
    print('running HS-preprocess')
    if esm_fasta is None:
        if path_exists(input_dir):
            get_single_chain(input_dir, output_dir)
            get_gpsite_split(output_dir, domain_out_dir)

    else:
        if path_exists(esm_fasta):
            print('===================running esmfold===================')
            subprocess.run(f"conda run -n surf2spot_tools esm-fold -i {esm_fasta} -o {input_dir} --cpu-only", shell=True)
            get_single_chain(input_dir, output_dir)
            get_gpsite_split(output_dir, domain_out_dir)
            

def run_HS_craft(
    output_dir: Path = typer.Option(..., "-i", help="输出目录"),
    gpsite_dir: Path = typer.Option(..., "-gpsite_dir", help="输出gpsite氨基酸得分目录"),
    seq_fasta: Path = typer.Option(..., "-s", help="输出的序列路径，作为prottrans的输入"),
    emb_out_path: Path = typer.Option(..., "-emb", "--emb_path", help="domain分割的最小长度阈值，默认为 30"),
    split_top_rate: float = typer.Option(0.66, "--split_top_rate", help="选top百分之多少的进行进一步筛选"),
    probe: float = typer.Option(5.0, "--probe", help="设置探针半径，默认为 5.0")
):
    print('running HS-craft')
    if path_exists(gpsite_dir) and path_exists(output_dir):
        extract_sequences_from_pdb(output_dir, seq_fasta)
        run_prottrans(seq_fasta, emb_out_path)
        surface_code = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/generate_surface.py')
        print('===================running surface calculation===================')
        subprocess.run(f"conda run -n surf2spot_tools python {surface_code} -i {output_dir} -o {output_dir} --probe {probe}", shell=True)
        atom_feature_engineering(output_dir)
        HS_surfpart(output_dir, gpsite_dir, split_top_rate)


def run_HS_predict(
    input_dir: Path = typer.Option(..., "-i", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", help="输出目录"),
    emb_path: Path = typer.Option(..., "-emb", "--emb_path", help="prottrans的输出"),
    model_path: Path = typer.Option(..., "--model", help="模型权重"),
    threshold: float = typer.Option(0.4,"--threshold", help="分割阈值") 
):
    print('running HS-predict')
    if path_exists(emb_path) and path_exists(input_dir) and path_exists(model_path):
        mk_dir(output_dir)
        HS_predict(input_dir, output_dir, emb_path, model_path, threshold)


def run_HS_draw(
    input_dir: Path = typer.Option(..., "-i", help="输入pdb文件的目录"),
    output_dir: Path = typer.Option(..., "-o", help="输出目录"),
):
    print('running HS-draw')
    if path_exists(input_dir) and path_exists(output_dir):
        HS_draw(input_dir, output_dir)


