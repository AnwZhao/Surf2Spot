import argparse
import csv
import hashlib
import logging
import os
import sys

import time
import pandas as pd
from pathlib import Path
from typing import List
from pymol import cmd
from biopandas.pdb import PandasPdb
from torch import compile as torch_compile
from Bio.PDB import PDBParser, Superimposer
import numpy as np

from Surf2Spot.data.chainsaw.src import constants, featurisers
from Surf2Spot.data.chainsaw.src.domain_assignment.util import convert_domain_dict_strings
from Surf2Spot.data.chainsaw.src.factories import pairwise_predictor
from Surf2Spot.data.chainsaw.src.models.results import PredictionResult
from Surf2Spot.data.chainsaw.src.prediction_result_file import PredictionResultsFile
from Surf2Spot.data.chainsaw.src.utils import common as common_utils
from Surf2Spot.data.chainsaw.src.utils.pymol_3d_visuals import generate_pymol_image


LOG = logging.getLogger(__name__)
OUTPUT_COLNAMES = ['chain_id', 'sequence_md5', 'nres', 'ndom', 'chopping', 'confidence', 'time_sec']
ACCEPTED_STRUCTURE_FILE_SUFFIXES = ['.pdb', '.cif']


def setup_logging():
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    # log all messages to stderr so results can be sent to stdout
    logging.basicConfig(level=loglevel,
                    stream=sys.stderr,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def get_input_method(args):
    number_of_input_methods = sum([ args.uniprot_id is not None,
                                    args.uniprot_id_list_file is not None,
                                    args.structure_directory is not None,
                                    args.structure_file is not None,
                                    args.pdb_id_list_file is not None,
                                    args.pdb_id is not None])
    if number_of_input_methods != 1:
        raise ValueError('Exactly one input method must be provided')
    if args.uniprot_id is not None:
        return 'uniprot_id'
    elif args.uniprot_id_list_file is not None:
        return 'uniprot_id_list_file'
    elif args.structure_directory is not None:
        return 'structure_directory'
    elif args.structure_file is not None:
        return 'structure_file'
    else:
        raise ValueError('No input method provided')

def load_model(*,
               model_dir: str,
               remove_disordered_domain_threshold: float = 0.35,
               min_ss_components: int = 2,
               min_domain_length: int = 30,
               post_process_domains: bool = True,):
    config = common_utils.load_json(os.path.join(model_dir, "config.json"))
    feature_config = common_utils.load_json(os.path.join(model_dir, "feature_config.json"))
    config["learner"]["remove_disordered_domain_threshold"] = remove_disordered_domain_threshold
    config["learner"]["post_process_domains"] = post_process_domains
    config["learner"]["min_ss_components"] = min_ss_components
    config["learner"]["min_domain_length"] = min_domain_length
    config["learner"]["dist_transform_type"] = config["data"].get("dist_transform", 'min_replace_inverse')
    config["learner"]["distance_denominator"] = config["data"].get("distance_denominator", None)
    learner = pairwise_predictor(config["learner"], output_dir=model_dir)
    learner.feature_config = feature_config
    learner.load_checkpoints()
    learner.eval()
    try:
        learner = torch_compile(learner)
    except:
        pass
    return learner


def predict(model, pdb_path, renumber_pdbs=True, pdbchain=None) -> List[PredictionResult]:
    """
    Makes the prediction and returns a list of PredictionResult objects
    """
    start = time.time()

    # get model structure metadata
    model_structure = featurisers.get_model_structure(pdb_path)

    if pdbchain is None:
        LOG.warning(f"No chain specified for {pdb_path}, using first chain")
        # get all the chain ids from the model structure
        all_chain_ids = [c.id for c in model_structure.get_chains()]
        # take the first chain id
        pdbchain = all_chain_ids[0]

    model_residues = featurisers.get_model_structure_residues(model_structure, chain=pdbchain)
    model_res_label_by_index = { int(r.index): str(r.res_label) for r in model_residues}
    model_structure_seq = "".join([r.aa for r in model_residues])
    model_structure_md5 = hashlib.md5(model_structure_seq.encode('utf-8')).hexdigest()

    x = featurisers.inference_time_create_features(pdb_path,
                                                    feature_config=model.feature_config,
                                                    chain=pdbchain,
                                                    renumber_pdbs=renumber_pdbs,
                                                    model_structure=model_structure,
                                                   )

    A_hat, domain_dict, confidence = model.predict(x)
    # Convert 0-indexed to 1-indexed to match AlphaFold indexing:
    domain_dict = [{k: [r + 1 for r in v] for k, v in d.items()} for d in domain_dict]
    names_str, bounds_str = convert_domain_dict_strings(domain_dict[0])
    confidence = confidence[0]

    if names_str == "":
        names = bounds = ()
    else:
        names = names_str.split('|')
        bounds = bounds_str.split('|')

    assert len(names) == len(bounds)

    class Seg:
        def __init__(self, domain_id: str, start_index: int, end_index: int):
            self.domain_id = domain_id
            self.start_index = int(start_index)
            self.end_index = int(end_index)
        
        def res_label_of_index(self, index: int):
            if index not in model_res_label_by_index:
                raise ValueError(f"Index {index} not in model_res_label_by_index ({model_res_label_by_index})")
            return model_res_label_by_index[int(index)]

        @property
        def start_label(self):
            return self.res_label_of_index(self.start_index)
        
        @property
        def end_label(self):
            return self.res_label_of_index(self.end_index)

    class Dom:
        def __init__(self, domain_id, segs: List[Seg] = None):
            self.domain_id = domain_id
            if segs is None:
                segs = []
            self.segs = segs

        def add_seg(self, seg: Seg):
            self.segs.append(seg)

    # gather choppings into segments in domains
    domains_by_domain_id = {}
    for domain_id, chopping_by_index in zip(names, bounds):
        if domain_id not in domains_by_domain_id:
            domains_by_domain_id[domain_id] = Dom(domain_id)
        start_index, end_index = chopping_by_index.split('-')
        seg = Seg(domain_id, start_index, end_index)
        domains_by_domain_id[domain_id].add_seg(seg)

    # sort domain choppings by the start residue in first segment
    domains = sorted(domains_by_domain_id.values(), key=lambda dom: dom.segs[0].start_index)

    # collect domain choppings as strings
    domain_choppings = []
    for dom in domains:
        # convert segments to strings
        segs_str = [f"{seg.start_label}-{seg.end_label}" for seg in dom.segs]
        segs_index_str = [f"{seg.start_index}-{seg.end_index}" for seg in dom.segs]
        LOG.info(f"Segments (index to label): {segs_index_str} -> {segs_str}")
        # join discontinuous segs with '_' 
        domain_choppings.append('_'.join(segs_str))

    # join domains with ','
    chopping_str = ','.join(domain_choppings)

    num_domains = len(domain_choppings)
    if num_domains == 0:
        chopping_str = None
    runtime = round(time.time() - start, 3)
    result = PredictionResult(
        pdb_path=pdb_path,
        sequence_md5=model_structure_md5,
        nres=len(model_structure_seq),
        ndom=num_domains,
        chopping=chopping_str,
        confidence=confidence,
        time_sec=runtime,
    )

    LOG.info(f"Runtime: {round(runtime, 3)}s")
    return result


def write_csv_results(csv_writer, prediction_results: List[PredictionResult]):
    """
    Render list of PredictionResult results to file pointer
    """
    for res in prediction_results:
        row = {
            'chain_id': res.chain_id,
            'sequence_md5': res.sequence_md5,
            'nres': res.nres,
            'ndom': res.ndom,
            'chopping': res.chopping if res.chopping is not None else 'NULL',
            'confidence': f'{res.confidence:.3g}' if res.confidence is not None else 'NULL',
            'time_sec': f'{res.time_sec}' if res.time_sec is not None else 'NULL',
        }
        csv_writer.writerow(row)


def get_csv_writer(file_pointer):
    csv_writer = csv.DictWriter(file_pointer,
                                fieldnames=OUTPUT_COLNAMES,
                                delimiter='\t')
    return csv_writer


def main(args):
    outer_save_dir = args.save_dir
    if args.use_first_chain:
        # use the first chain in the PDB file
        pdb_chain_id = None
    else:
        pdb_chain_id = 'A'

    input_method = get_input_method(args)
    model = load_model(
        model_dir=args.model_dir,
        remove_disordered_domain_threshold=args.remove_disordered_domain_threshold,
        min_ss_components=args.min_ss_components,
        min_domain_length=args.min_domain_length,
        post_process_domains=args.post_process_domains,
    )
    os.makedirs(outer_save_dir, exist_ok=True)
    output_path = Path(args.output).absolute()

    prediction_results_file = PredictionResultsFile(
        csv_path=output_path,
        # use args.allow_append to mean allow_skip and allow_append
        allow_append=args.allow_append,
        allow_skip=args.allow_append,
    )

    if input_method == 'structure_directory':
        structure_dir = args.structure_directory
        for idx, fname in enumerate(os.listdir(structure_dir)):
            suffix = Path(fname).suffix
            LOG.debug(f"Checking file {fname} (suffix: {suffix}) ..")
            if suffix not in ACCEPTED_STRUCTURE_FILE_SUFFIXES:
                continue

            chain_id = Path(fname).stem
            result_exists = prediction_results_file.has_result_for_chain_id(chain_id)
            if result_exists:
                LOG.info(f"Skipping file {fname} (result for '{chain_id}' already exists)")
                continue

            pdb_path = os.path.join(structure_dir, fname)
            LOG.info(f"Making prediction for file {fname} (chain '{chain_id}')")
            result = predict(model, pdb_path, pdbchain=pdb_chain_id, renumber_pdbs=args.renumber_pdbs)
            prediction_results_file.add_result(result)
            if args.pymol_visual:
                generate_pymol_image(
                    pdb_path=str(result.pdb_path),
                    chopping=result.chopping or '',
                    image_out_path=os.path.join(str(outer_save_dir), f'{result.pdb_path.name.replace(".pdb", "")}.png'),
                    path_to_script=os.path.join(str(outer_save_dir), 'image_gen.pml'),
                    pymol_executable=constants.PYMOL_EXE,
                )
    elif input_method == 'structure_file':
        result = predict(model, args.structure_file, pdbchain=pdb_chain_id)
        prediction_results_file.add_result(result)
        if args.pymol_visual:
            generate_pymol_image(
                pdb_path=str(result.pdb_path),
                chopping=result.chopping or '',
                image_out_path=os.path.join(str(outer_save_dir), f'{result.pdb_path.name.replace(".pdb", "")}.png'),
                path_to_script=os.path.join(str(outer_save_dir), 'image_gen.pml'),
                pymol_executable=constants.PYMOL_EXE,
            )
    else:
        raise NotImplementedError('Not implemented yet')

    prediction_results_file.flush()
    LOG.info("DONE")



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


def pdb_domain_resi(pdb_path, pdb_name, Chainsaw_out_path):
    pdb_top_resi_dict = {}
    df = pd.read_csv(Chainsaw_out_path, sep='\t', header=0, index_col='chain_id')
    domain_str = list(df.loc[[pdb_name]]['chopping'])[0]
    pdb_file = os.path.join(pdb_path, pdb_name+'.pdb')
    domain_split = domain_str.split(',')  #14-149_290-412,152-287_422-450
    for i in range(len(domain_split)):
        domain = domain_split[i]
        domain_range = get_split_range(domain)
        pdb_top_resi_dict[pdb_name+'_domain'+str(i)] = {'domain_resi':[str(i) for i in domain_range], 'domain_range':domain}
            
    return pdb_top_resi_dict


def get_domain_split(structure_directory, output_file, min_domain_length):
    """直接返回一个预定义的 args 对象，而不从命令行解析"""
    args = argparse.Namespace(
        model_dir=f'{constants.REPO_ROOT}/saved_models/model_v3',
        output=output_file,  # required=True 需手动设置
        uniprot_id=None,
        uniprot_id_list_file=None,
        structure_directory=structure_directory,
        structure_file=None,
        allow_append=False,  # --append
        pdb_id=None,
        pdb_id_list_file=None,
        save_dir='results',
        post_process_domains=True,  # --no_post_processing 会设为 False
        remove_disordered_domain_threshold=0.35,
        min_domain_length=min_domain_length,   #30
        min_ss_components=2,
        pymol_visual=False,  # --pymol_visual 会设为 True
        use_first_chain=True,  # 默认 True
        renumber_pdbs=False,   # --renumber_pdbs 会设为 True
    )
    main(args)



def extract_chain_from_pdb(input_pdb, output_pdb, chain_id):
    """
    使用 BioPandas 从 PDB 文件中提取指定链并保存为新文件
    
    参数:
        input_pdb (str): 输入的 PDB 文件路径
        output_pdb (str): 输出的 PDB 文件路径
        chain_id (str): 要提取的链 ID (如 "A", "B" 等)
    """
    # 读取 PDB 文件
    ppdb = PandasPdb()
    ppdb.read_pdb(input_pdb)
    
    # 提取目标链的原子数据
    atom_df = ppdb.df['ATOM']
    chain_atoms = atom_df[atom_df['chain_id'] == chain_id]
    
    # 创建一个新的 PandasPdb 对象并填充数据
    new_ppdb = PandasPdb()
    new_ppdb.df['ATOM'] = chain_atoms
    
    # 保存到新文件
    new_ppdb.to_pdb(path=output_pdb, records=['ATOM'], gz=False)  # 不保存 HETATM/ANISOU 等
   


def align_rmsd(proteinA_pdb, proteinB_pdb, chain, resi_list):
    # 初始化 PyMOL
    cmd.reinitialize()
    
    # 加载 PDB 文件
    cmd.load(proteinA_pdb, "proteinA")
    cmd.load(proteinB_pdb, "proteinB")
    resi_select = 'resi '+' + resi '.join(resi_list)

    # 进行结构对齐 (将 proteinB 对齐到 proteinA)
    rmsd = cmd.align(f"proteinB and chain {chain} and ({resi_select})", f"proteinA and chain {chain}")
    return rmsd[0], rmsd[3]


def align_ligand_rmsd(proteinA_pdb, proteinB_pdb, chain):
    # 初始化 PyMOL
    cmd.reinitialize()
    
    # 加载 PDB 文件
    cmd.load(proteinA_pdb, "proteinA")
    cmd.load(proteinB_pdb, "proteinB")
    cmd.remove(f'chain {chain}')
    cmd.select('B_inter', f"proteinB and not chain {chain}")

    # 进行结构对齐 (将 proteinB 对齐到 proteinA)
    rmsd = cmd.align("B_inter", "proteinA")
    return rmsd[0], rmsd[3]
    





def get_chain_coords(pdb_file, chain_id):
    """
    使用 BioPandas 提取指定链的坐标（优先从 ATOM 表查找，找不到则查 HETATM 表）
    
    返回:
        np.ndarray: 原子坐标数组 (N,3)
        list: 对应的元素符号列表
    """
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_file)
    
    # 优先从 ATOM 表查找
    if 'ATOM' in ppdb.df:
        atom_df = ppdb.df['ATOM']
        chain_atoms = atom_df[atom_df['chain_id'] == chain_id]
        if len(chain_atoms) > 0:
            coords = chain_atoms[['x_coord', 'y_coord', 'z_coord']].values
            elements = chain_atoms['element_symbol'].tolist()
            return coords, elements
    
    # 如果 ATOM 表没有，则从 HETATM 表查找
    if 'HETATM' in ppdb.df:
        hetatm_df = ppdb.df['HETATM']
        chain_hetatm = hetatm_df[hetatm_df['chain_id'] == chain_id]
        if len(chain_hetatm) > 0:
            coords = chain_hetatm[['x_coord', 'y_coord', 'z_coord']].values
            elements = chain_hetatm['element_symbol'].tolist()
            return coords, elements
    
    raise ValueError(f"Chain {chain_id} not found in ATOM or HETATM records")
    

def calculate_ligand_rmsd_after(pdb_file1, pdb_file2, chain_id):
    """
    使用 BioPandas 计算两个 PDB 文件指定链的 RMSD
    """
    # 获取两个结构的链坐标和元素
    coords1, elements1 = get_chain_coords(pdb_file1, chain_id)
    coords2, elements2 = get_chain_coords(pdb_file2, chain_id)
    
    # 检查原子数量和元素是否匹配
    if len(coords1) != len(coords2):
        raise ValueError(f"Atom count mismatch: {len(coords1)} vs {len(coords2)}")
    
    if elements1 != elements2:
        raise ValueError("Element types do not match between chains")
    
    # 计算 RMSD（需要先叠加）
    def kabsch_rmsd(coords1, coords2):
        """Kabsch 算法计算最优叠加后的 RMSD"""
        centroid1 = coords1.mean(axis=0)
        centroid2 = coords2.mean(axis=0)
        coords1 -= centroid1
        coords2 -= centroid2
        H = coords2.T @ coords1
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = Vt.T @ U.T
        rotated = coords2 @ R
        return np.sqrt(np.mean(np.sum((coords1 - rotated)**2, axis=1)))
    
    rmsd = kabsch_rmsd(coords1, coords2)
    #print(f"RMSD of Chain {chain_id}: {rmsd:.3f} Å")
    return rmsd



def calculate_ligand_rmsd_before(pdb_file1, pdb_file2, chain_id):
    """
    直接计算两个链的坐标 RMSD（不进行叠加）
    """
    # 获取两个结构的链坐标和元素
    coords1, elements1 = get_chain_coords(pdb_file1, chain_id)
    coords2, elements2 = get_chain_coords(pdb_file2, chain_id)
    
    # 检查原子数量和元素是否匹配
    if len(coords1) != len(coords2):
        raise ValueError(f"Atom count mismatch: {len(coords1)} vs {len(coords2)}")
    
    if elements1 != elements2:
        raise ValueError("Element types do not match between chains")
    
    # 直接计算 RMSD（无叠加）
    squared_dist = np.sum((coords1 - coords2)**2, axis=1)
    rmsd = np.sqrt(np.mean(squared_dist))
    
    #print(f"Direct RMSD of Chain {chain_id}: {rmsd:.3f} Å (without superposition)")
    return rmsd



def get_all_chain_ids(pdb_file):
    """使用 Biopython 获取 PDB 文件中所有链 ID"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_file)
    
    chain_ids = []
    
    for model in structure:
        for chain in model:
            chain_ids.append(chain.id)
    
    # 去重并排序
    return sorted(set(chain_ids))
    
'''
def main():
    parser = argparse.ArgumentParser(description='input parse')
    parser.add_argument('-i','--input_dir', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')
    parser.add_argument('-o','--chainsaw_split', type=str, metavar='', required= True,
                        help='(Please enter an absolute path)')  # .tsv
    args = parser.parse_args()

    pdb_path = args.input_dir
    Chainsaw_out_path = args.chainsaw_split
    
    get_domain_split(pdb_path, Chainsaw_out_path)
'''

def get_domain_tsv(output_dir, domain_out_tsv, min_domain_length):

    pdb_path = output_dir
    Chainsaw_out_path = domain_out_tsv
    
    get_domain_split(pdb_path, Chainsaw_out_path, min_domain_length)

