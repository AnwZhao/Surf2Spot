from pathlib import Path
import os


REPO_ROOT = Path(__file__).parent.parent.resolve()
STRIDE_EXE = os.environ.get('STRIDE_EXE', str(REPO_ROOT / "stride" / "stride"))
PYMOL_EXE = "/Applications/PyMOL.app/Contents/MacOS/PyMOL" # only required if you want to generate 3D images

_3to1 = {
'ALA': 'A',
'CYS': 'C',
'ASP': 'D',
'GLU': 'E',
'PHE': 'F',
'GLY': 'G',
'HIS': 'H',
'ILE': 'I',
'LYS': 'K',
'LEU': 'L',
'MET': 'M',
'ASN': 'N',
'PRO': 'P',
'GLN': 'Q',
'ARG': 'R',
'SER': 'S',
'THR': 'T',
'VAL': 'V',
'TRP': 'W',
'TYR': 'Y',
}

