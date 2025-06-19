import numpy as np
from tqdm import tqdm
import os, argparse, datetime, itertools

import torch
import torch_geometric
from biopandas.pdb import PandasPdb
from torch_geometric.loader import DataLoader

from Surf2Spot.data.feature_extraction.ProtTrans import get_ProtTrans
from Surf2Spot.data.feature_extraction.process_structure import get_pdb_xyz, process_dssp, match_dssp
from Surf2Spot.data.utils import *
from Surf2Spot.data.model import *
from Bio import SeqIO
from Bio.SeqUtils import IUPACData
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.PDB import StructureBuilder, PDBParser, Selection, PDBIO, parse_pdb_header
from Bio.PDB.PDBIO import Select
PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]

############ Set to your own path! ############
script_path = os.path.dirname(os.path.abspath(__file__))
ProtTrans_path = os.path.join(script_path, "model_emb/prot_t5_xl_half_uniref50-enc")
model_path = os.path.join(script_path, "model_gpsite")


class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"



def longLetterToNormalLetter(letter):
    NTP = ["ATP", "GTP", "CTP", "TTP", "UTP"]
    if "A" in letter:
        return "A"
    elif "C" in letter:
        return "C"
    elif "T" in letter and "TM" not in letter and "TP" not in letter:
        return "T"
    elif "G" in letter:
        return "G"
    elif "U" in letter and "UNK" not in letter:
        return "U"
    elif "UNK" in letter:
        return "N"
    elif "I" in letter:
        return "I"
    elif "N" in letter:
        return "N"
    elif "P" in letter:
        return "P"
    elif "TM" in letter:
        return "N"
    elif "ATP" in letter:
        return "A"
    elif "GTP" in letter:
        return "G"
    elif "CTP" in letter:
        return "C"
    elif "TTP" in letter:
        return "T"
    elif "UTP" in letter:
        return "U"
    elif "R" in letter:
        return "R"
    else:
        return "N"


def getProteinFastaFromPdb(pdb_file):
    ppdb = PandasPdb()
    pdbStruc = ppdb.read_pdb(pdb_file)
    seqInfo = pdbStruc.df["OTHERS"].loc[pdbStruc.df["OTHERS"].record_name == "SEQRES"]
    if not seqInfo.empty:
        chain_dict = dict([(l[5], []) for l in seqInfo.entry])
        for c in list(chain_dict.keys()):
            chain_seq = [l[13:].split() for l in seqInfo.entry if l[5] == c]
            for x in chain_seq:
                chain_dict[c].extend(x)
    else:
        chain_dict = {}

    chains = chain_dict.keys()
    pChains = []
    naChains = []
    naType = {}
    for c in chains:
        sequence = chain_dict[c]
        if len(set(sequence) & set(PROTEIN_LETTERS)) > 0:
            pChains.append(c)
        else:
            sequence = [longLetterToNormalLetter(i) for i in sequence]
            if len(set(sequence)) == 1 and list(set(sequence))[0] == "N": continue
            if "T" in "".join(sequence) or "D" in "".join(sequence):
                naType[c] = "DNA"
                naChains.append(c)
            else:
                naType[c] = "RNA"
                naChains.append(c)

    pChain2seq = {}
    dnaChain2Seq = {}
    maxLenPChain = []
    for i, j in itertools.product(pChains, naChains):
        pSeq = chain_dict[i]
        tmpSeq = []
        for x in pSeq:
            tmpSeq.append(protein_letters_3to1[x] if x in protein_letters_3to1 else "X")
        # pSeq = [protein_letters_3to1[x] for x in pSeq if x in protein_letters_3to1]
        pSeq = "".join(tmpSeq)
        naSeq = chain_dict[j]
        naSeq = [longLetterToNormalLetter(x) for x in naSeq]
        naSeq = "".join(naSeq)
        pChain2seq.update({i: pSeq})
        dnaChain2Seq.update({j: naSeq})
        if not maxLenPChain:
            maxLenPChain = [i, len(pSeq)]
        else:
            if len(pSeq) > maxLenPChain[1]:
                maxLenPChain = [i, len(pSeq)]
    return pChain2seq, maxLenPChain


def getProteinFastaFromPdb1(pdb_file):
    pChain2seq = {}
    maxLenPChain = []
    with open(pdb_file, 'r') as pdb_file:
        for record in SeqIO.parse(pdb_file, 'pdb-atom'):
            chain_id = record.id.split(":")[1]
            pSeq = str(record.seq)
            pChain2seq.update({chain_id: pSeq})
            if not maxLenPChain:
                maxLenPChain = [chain_id, len(pSeq)]
            else:
                if len(pSeq) > maxLenPChain[1]:
                    maxLenPChain = [chain_id, len(pSeq)]
    return pChain2seq, maxLenPChain

def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


def extractPDB(infilename, outfilename, chainIds=None):
    """
    extractPDB: Extract selected chains from a PDB and save the extracted chains to an output file.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
    """
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if chainIds == None or chain.get_id() in chainIds:
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered(), preserve_atom_numbering=True)


def process_pdbs(pdb_list_file, chain_id, is_pdb_list, outpath):
    ID_list = []
    seq_list = []

    if is_pdb_list:
        with open(pdb_list_file) as f:
            for line in f.readlines():
                line_fields = line.strip().split("\t")
                pdb_file = line_fields[0]
                pdb_id = pdb_file.split("/")[-1].split(".")[0]
                pChain2Seq, maxLenPChain = getProteinFastaFromPdb1(pdb_file)
                if len(line_fields) == 2:
                    chain_id = line_fields[1]
                else:
                    chain_id = maxLenPChain[0]

                pSeq = pChain2Seq[chain_id]
                pSeq = pSeq.upper()
                pSeq = remove_non_standard_aa(pSeq)
                pSeq = pSeq[0:min(MAX_SEQ_LEN, len(pSeq))]

                sub_pdb_file = path_join([outpath, "pdb", "{}.pdb".format(pdb_id)])
                extractPDB(pdb_file, sub_pdb_file, [chain_id])
                ID_list.append(pdb_id)
                seq_list.append(pSeq)
                cmd = "echo 'time | INFO | root | Predicted structure for {} with length 0, pLDDT 0, pTM 0 in 0s. 1 / 1 completed.' > {}/esmfold_pred.log".format(pdb_id, outpath)
                os.system(cmd)
    else:
        pdb_id = pdb_list_file.split("/")[-1].split(".")[0]
        pChain2Seq, maxLenPChain = getProteinFastaFromPdb1(pdb_list_file)
        if not chain_id:
            chain_id = maxLenPChain[0]
        pSeq = pChain2Seq[chain_id]

        sub_pdb_file = path_join([outpath, "pdb", "{}.pdb".format(pdb_id)])
        extractPDB(pdb_list_file, sub_pdb_file, [chain_id])
        ID_list.append(pdb_id)
        seq_list.append(pSeq)
        cmd = "echo 'time | INFO | root | Predicted structure for {} with length 0, pLDDT 0, pTM 0 in 0s. 1 / 1 completed.' > {}/esmfold_pred.log".format(pdb_id, outpath)
        os.system(cmd)

    if len(ID_list) == len(seq_list):
        if len(ID_list) > MAX_INPUT_SEQ:
            return 1
        else:
            new_fasta = "" # with processed IDs and seqs
            for i in range(len(ID_list)):
                new_fasta += (">" + ID_list[i] + "\n" + seq_list[i] + "\n")
            with open(path_join([outpath, "test_seq.fa"]), "w") as f:
                f.write(new_fasta)
            return [ID_list, seq_list]
    else:
        return -1


def makeLink(sourcePath, targetPath):
    if os.path.islink(targetPath) or os.path.exists(targetPath):
        os.remove(targetPath)
    os.symlink(sourcePath, targetPath)


def link_pdb_to_target_dir(pdb_list_file, is_pdb_list, target_dir):
    if is_pdb_list:
        with open(pdb_list_file) as f:
            for line in f.readlines():
                line_fields = line.strip().split("\t")
                pdb_file = line_fields[0]
                target_pdb_path = path_join([target_dir, pdb_file])
                makeLink(pdb_file, target_pdb_path)
    else:
        target_pdb_path = path_join([target_dir, pdb_list_file])
        makeLink(os.path.abspath(pdb_list_file), target_pdb_path)


def extract_feat(ID_list, seq_list, outpath, gpu, prottrans_model, prottrans_tokenizer):

    Min_protrans = torch.tensor(np.load(os.path.join(script_path, "feature_extraction/Min_ProtTrans_repr.npy")), dtype = torch.float32)
    Max_protrans = torch.tensor(np.load(os.path.join(script_path, "feature_extraction/Max_ProtTrans_repr.npy")), dtype = torch.float32)
    get_ProtTrans(ID_list, seq_list, Min_protrans, Max_protrans, ProtTrans_path, outpath, gpu, prottrans_model, prottrans_tokenizer)

    print("Processing PDB files...")
    for ID in tqdm(ID_list):
        with open(path_join([outpath, "pdb", ID + ".pdb"]), "r") as f:
            X = get_pdb_xyz(f.readlines()) # [L, 5, 3]
        torch.save(torch.tensor(X, dtype = torch.float32), path_join([outpath, "pdb", ID + '.tensor']))

    print("Extracting DSSP features...")
    for i in tqdm(range(len(ID_list))):
        ID = ID_list[i]
        seq = seq_list[i]

        os.system("{}/feature_extraction/mkdssp -i {}/pdb/{}.pdb -o {}/DSSP/{}.dssp".format(script_path, outpath, ID, outpath, ID))
        dssp_seq, dssp_matrix = process_dssp("{}/DSSP/{}.dssp".format(outpath, ID))
        if dssp_seq != seq:
            dssp_matrix = match_dssp(dssp_seq, dssp_matrix, seq)

        torch.save(torch.tensor(np.array(dssp_matrix), dtype = torch.float32), "{}/DSSP/{}.tensor".format(outpath, ID))
        #os.system("rm {}/DSSP/{}.dssp".format(outpath, ID))


def predict(ID_list, outpath, batch, gpu):
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')

    node_input_dim = nn_config['node_input_dim']
    edge_input_dim = nn_config['edge_input_dim']
    hidden_dim = nn_config['hidden_dim']
    layer = nn_config['layer']
    augment_eps = nn_config['augment_eps']
    dropout = nn_config['dropout']

    task_list = ["PRO", "PEP", "DNA", "RNA", "ZN", "CA", "MG", "MN", "ATP", "HEME"]

    # Test
    test_dataset = ProteinGraphDataset(ID_list, outpath)
    test_dataloader = DataLoader(test_dataset, batch_size = batch, shuffle=False, drop_last=False, num_workers=8, prefetch_factor=2)

    models = []
    for fold in range(5):
        state_dict = torch.load(os.path.join(model_path, 'fold%s.ckpt'%fold), device)
        model = GPSite(node_input_dim, edge_input_dim, hidden_dim, layer, augment_eps, dropout, task_list).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    test_pred_dict = {}
    for data in tqdm(test_dataloader):
        data = data.to(device)

        with torch.no_grad():
            print('---------------', data.X.shape,  data.node_feat.shape, data.edge_index.shape)
            outputs = [model(data.X, data.node_feat, data.edge_index, data.batch).sigmoid() for model in models]
            outputs = torch.stack(outputs,0).mean(0) # average the predictions from 5 models

        IDs = data.name
        outputs_split = torch_geometric.utils.unbatch(outputs, data.batch)
        for i, ID in enumerate(IDs):
            test_pred_dict[ID] = []
            for j in range(len(task_list)):
                test_pred_dict[ID].append(list(outputs_split[i][:,j].detach().cpu().numpy()))

    return test_pred_dict


def main(pdb_file, is_pdb_list, seq_info, outpath, batch, gpu, prottrans_model, prottrans_tokenizer):
    ID_list, seq_list = seq_info
    # for dir_name in ["pdb", "ProtTrans", "DSSP", "pred"]:
    #     os.makedirs(path_join([outpath, dir_name]), exist_ok=True)

    # link_pdb_to_target_dir(pdb_file, is_pdb_list, path_join([outpath, "pdb"]))
    print("\n######## Feature extraction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))
    extract_feat(ID_list, seq_list, outpath, gpu, prottrans_model, prottrans_tokenizer)

    print("\n######## Prediction begins at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    predictions = predict(ID_list, outpath, batch, gpu)

    print("\n######## Prediction is done at {}. ########\n".format(datetime.datetime.now().strftime("%m-%d %H:%M")))

    export_predictions(predictions, seq_list, outpath)

    print("\n######## Results are saved in {} ########\n".format(outpath + "pred/"))


def run_gpsite(pdb_file, outpath, gpu, prottrans_model, prottrans_tokenizer, is_pdb_list=False, batch=4):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # run_id = args.pdb_list.split("/")[-1].split(".")[0].replace(" ", "_")
    outpath = os.path.join(outpath, timestamp)+'/'
    os.makedirs(outpath, exist_ok = True)
    for dir_name in ["pdb", "ProtTrans", "DSSP", "pred"]:
        os.makedirs(path_join([outpath, dir_name]), exist_ok=True)

    seq_info = process_pdbs(pdb_file, None, is_pdb_list, outpath)

    if seq_info == -1:
        print("The format of your input fasta file is incorrect! Please check!")
    elif seq_info == 1:
        print("Too much sequences! Up to {} sequences are supported each time!".format(MAX_INPUT_SEQ))
    else:
        # pass
        main(pdb_file, is_pdb_list, seq_info, outpath, batch, gpu, prottrans_model, prottrans_tokenizer)

