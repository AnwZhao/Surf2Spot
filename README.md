# Surf2Spot: A Geometric Model for Prediction of Epitope and Binding Sites on Target Protein

Surf2Spot is a protein binding site prediction software based on deep learning, which is specially designed to identify protein-protein interaction hotspots (PPI hotspots) and nano-antibody-antigen binding sites (NAI epitopes). The software innovatively combines protein surface point cloud data with dynamic graph convolutional neural network (DGCNN), and accurately predicts the binding sites by uniformly sampling protein surface and coding its spatial and physical and chemical properties. 
Surf2Spot adopts point cloud modeling strategy to effectively identify solvent exposed residues and filter out buried residues, thus improving prediction accuracy and optimizing calculation efficiency. This tool is dedicated to providing theoretical support and target screening basis for rational design of protein binding molecules and nano-antibodies.


## Software prerequisites
Surf2Spot relies on external software/libraries to handle protein databank files and surface files, to compute chemical/geometric features and coordinates, and to perform neural network calculations. The following is the list of required libraries and programs, as well as the version on which it was tested (in parenthesis).

* [msms(2.6.1)](https://ccsb.scripps.edu/msms/)
* [apbs(3.4.1)](https://github.com/Electrostatics/apbs-pdb2pqr/releases)
* [pdb2pqr(2.1.1)](https://github.com/Electrostatics/pdb2pqr/releases?page=2)


## Installation
1. Clone the package
   ```shell
   git clone https://github.com/AnwZhao/Surf2Spot
   cd Surf2Spot
    ```
    Download the msms(2.6.1), apbs(3.4.1), pdb2pqr(2.1.1) package to `./Surf2Spot/data/masif/extra_dependence` and install.

   If the version used or directory name is not corresponding, please modify the directory name of the path in `./Surf2Spot/data/generate_surface.py`

   ```python
   os.environ["LD_LIBRARY_PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'masif/extra_dependence/APBS-3.4.1/lib')  
   msms_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'masif/extra_dependence/msms/msms.x86_64Linux2.2.6.1')  
   apbs_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),  'masif/extra_dependence/APBS-3.4.1/bin/apbs')  
   pdb2pqr_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "masif/extra_dependence/pdb2pqr/pdb2pqr")  
   multivalue_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)),  "masif/extra_dependence/APBS-3.4.1/share/apbs/tools/bin/multivalue")
   ```  
    
    Run the code to download the model of support software.
    ```shell
    bash download.sh
    ```
    
    
2. Prepare the environment
We recommend using conda to substantially facilitate installation of all Python dependencies. 
Cuda version 11.8 and above is required. 
In order to avoid environmental conflicts, two different conda environments need to be built. Considering the version problems of some installation packages, please install python versions 3.7 and 3.10 respectively.
   ```shell
    conda  create  -n  surf2spot_tools  python=3.7    
    pip  install  "fair-esm[esmfold]"  
    pip  install  git+https://github.com/facebookresearch/esm.git  
    pip  install  'dllogger @ git+https://github.com/NVIDIA/dllogger.git'  
    pip  install  'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'   
    conda  install  -c  conda-forge  pymesh2  
    pip  install  scipy  biopython  rdkit  plyfile  IPython  joblib  
    pip  install  https://github.com/PyMesh/PyMesh/releases/download/v0.3/pymesh2-0.3-cp37-cp37m-linux_x86_64.whl
   ```

   ```shell
    conda  create  -n  surf2spot  python=3.9  
    pip  install  torch==2.1.1  torchvision==0.16.1  torchaudio==2.1.1  --index-url  https://download.pytorch.org/whl/cu118  
    conda  install  -c  conda-forge  pymol-open-source=3.0.0  
    pip  install  https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_cluster-1.6.3%2Bpt21cu118-cp39-cp39-linux_x86_64.whl  
    pip  install  https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_scatter-2.1.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl  
    pip  install  https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_sparse-0.6.18%2Bpt21cu118-cp39-cp39-linux_x86_64.whl  
    pip  install  https://data.pyg.org/whl/torch-2.1.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt21cu118-cp39-cp39-linux_x86_64.whl  
    pip  install  typer  
  
    pip  install  -e  .
   ```


## Model Weight

 Our trained models can be downloaded at [Surf2Spot_NB](https://huggingface.co/anwzhao/Surf2Spot_NB ) and [Surf2Spot_HS](https://huggingface.co/anwzhao/Surf2Spot_HS ).

Download the model weight `model.pt` of Surf2Spot  and put it in the ./model.


## Usage
### NAI epitopes prediction

```mermaid
graph LR
A[NB-preprocess] --> B[NB-craft] 
B --> C[NB-predict]
C --> D[NB-draw]
```
   ```shell
   conda  activate  surf2spot
   ```

1. Antigen preprocessing
   Simply put the pdb files for analysis in the `test_NB/input` folder.
   ```shell
    Surf2Spot  NB-preprocess  -i  test_NB/input  -o  test_NB/output  -ds  test_NB/chainsaw.tsv 
   ```
   If you only have the protein sequence, you only need to write the sequence into fasta file, and the model will automatically call esmfold to predict the structure.
   ```shell
    Surf2Spot  NB-preprocess  --esm  test_NB/esm.fasta  -i  test_NB/input  -o  test_NB/output  -ds  test_NB/chainsaw.tsv 
   ```

2. Feature engineering
    This step generates the characteristics of amino acids on the protein, including the embedding of prot5 and the domain partition results.
   ```shell
   Surf2Spot  NB-craft  -i  test_NB/output  -s  test_NB/seq.fasta  -ds  test_NB/chainsaw.tsv  -emb  test_NB/seq_prottrans.h5    
   ```

3. Model prediction
    Perform model prediction and output point cloud prediction results of NAI epitopes.
   ```shell
   Surf2Spot  NB-predict  -i  test_NB/output  -o  test_NB/predict  -emb  test_NB/seq_prottrans.h5  --model  model/NB/model.pt    
   ```

4. Result rendering and cluster analysis
    In this step, the predicted results of protein surface point cloud are mapped to amino acids, and the amino acids predicted as NAI epitopes are clustered to output the amino acid clusters that are finally suitable for designing nanobody.
   ```shell
   Surf2Spot  NB-draw  -i  test_NB/output  -o  test_NB/predict    
   ```
   
---
### PPI hotspots prediction

```mermaid
graph LR
A[HS-preprocess] --> B[HS-craft] 
B --> C[HS-predict]
C --> D[HS-draw]
```

   ```shell
   conda  activate  surf2spot
   ```

1. target protein preprocessing
    Simply put the pdb files for analysis in the `test_HS/input` folder.
   ```shell
    Surf2Spot  HS-preprocess  -i  test_HS/input  -o  test_HS/output  -gpsite_dir  test_HS/GPscore 
   ```
    If you only have the protein sequence, you only need to write the sequence into fasta file, and the model will automatically call esmfold to predict the structure. 
   ```shell
    Surf2Spot  HS-preprocess  --esm  test_HS/esm.fasta  -i  test_HS/input  -o  test_HS/output  -gpsite_dir  test_HS/GPscore 
   ```
   
2. Feature engineering
    This step generates the characteristics of amino acids on the protein, including the embedding of prot5 and the PPI probabilistic enrichment site by GPSite.
   ```shell
    Surf2Spot  HS-craft  -i  test_HS/output  -s  test_HS/seq.fasta  -emb  test_HS/seq_prottrans.h5  -gpsite_dir  test_HS/GPscore 
   ```

3. Model prediction
    Perform model prediction and output point cloud prediction results of PPI hotspots.
   ```shell
    Surf2Spot  HS-predict  -i  test_HS/output  -o  test_HS/predict  -emb  test_HS/seq_prottrans.h5  --model  model/HS/model.pt 
   ```

4. Result rendering and cluster analysis
    In this step, the predicted results of protein surface point cloud are mapped to amino acids, and the amino acids predicted as PPI hotspots are clustered to output the amino acid clusters that are finally suitable for designing nanobody.
   ```shell
   Surf2Spot  HS-draw  -i  test_HS/output  -o  test_HS/predict    
   ```
   *.csv and *_pre.pse are prediction result files, while *_cluster.pse represents the clustering of the prediction results. The selection of design anchors and hotspot residues can be guided by the clustering information in the *_cluster.pse file.

 
## Data
We prepared the corresponding target protein/antigen id and the [original pdb file](https://huggingface.co/datasets/anwzhao/Surf2Spot_raw_data).
If it's inconvenient for you to preprocess the data, we also provided the [processed data](https://huggingface.co/datasets/anwzhao/Surf2Spot_data).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                


## Citing this work

```bibtex

```

