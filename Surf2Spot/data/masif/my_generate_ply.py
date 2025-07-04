import os
import shutil
import numpy as np
import subprocess
import pymesh
import tempfile, shutil
#import Bio.PDB
from Bio.PDB import PDBParser, PDBIO, Select 
from Bio.PDB import NeighborSearch, Selection
from rdkit import Chem
from scipy.spatial import distance, KDTree
from IPython.utils import io
from joblib import Parallel, delayed
import sys
sys.path.append(os.path.dirname(__file__))
from compute_normal import compute_normal
from computeAPBS import computeAPBS
from computeCharges import computeCharges, assignChargesToNewMesh
from computeHydrophobicity import computeHydrophobicity
from computeMSMS import computeMSMS
from fixmesh import fix_mesh
from save_ply import save_ply

#from pymol import cmd
#import open3d as o3d
import json
from scipy.misc import derivative


def compute_inp_surface(prot_path, msms_bin,apbs_bin,pdb2pqr_bin,multivalue_bin,
                        prot_chain='A',
                        outdir='.',
                        probe = '5.0',
                        dist_threshold=5.0,
                        use_hbond=True,
                        use_hphob=True,
                        use_apbs=True,
                        compute_iface=True,
                        mesh_res=1.5,
                        epsilon=1.0e-6,
                        feature_interpolation=True,                        
                        out_name= None):    
    workdir = tempfile.mkdtemp()
    protname = os.path.basename(prot_path).replace(".pdb","")
    
    #mol = Chem.MolFromPDBBlock(prot_path)
    mol = Chem.MolFromPDBFile(prot_path, sanitize=False)
    if mol==None:
        mol = Chem.MolFromPDBBlock(prot_path, sanitize=False)
    atomCoords = mol.GetConformers()[0].GetPositions()
    print('len(atomCoords)',len(atomCoords))
    
    
    # Read protein and select aminino acids in the binding pocket
    parser = PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.
    
    structures = parser.get_structure('target', prot_path)
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    
    atoms  = Selection.unfold_entities(structure, 'A')
    ns = NeighborSearch(atoms)
    print('len(atoms)',len(atoms))
    
   
    #atoms_another  = Selection.unfold_entities(structure, 'B')
    
    close_residues= []
    for a in atomCoords:  
        close_residues.extend(ns.search(a, dist_threshold, level='R'))
    close_residues = Selection.uniqueify(close_residues)

    class SelectNeighbors(Select):
        def accept_residue(self, residue):
            #if residue in close_residues :
            #if residue in close_residues and residue.get_parent().id =='A':
            if residue in close_residues and residue.get_parent().id ==prot_chain:
                if all(a in [i.get_name() for i in residue.get_unpacked_list()] for a in ['N', 'CA', 'C', 'O']) or residue.resname=='HOH':
                    return True
                else:
                    return False
            else:
                return False

    pdbio = PDBIO()
    pdbio.set_structure(structure)
    print('SelectNeighbors()',SelectNeighbors())
    pdbio.save("%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold), SelectNeighbors())
     
    
    # Identify closes atom to the ligand
    structures = parser.get_structure('target', "%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold))
    structure = structures[0] # 'structures' may contain several proteins in this case only one.
    atoms = Selection.unfold_entities(structure, 'A')
    
    try:
        dist = [distance.euclidean(atomCoords.mean(axis=0), a.get_coord()) for a in atoms]
        atom_idx = np.argmin(dist)
        vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                    protonate=True, 
                                                                    one_cavity=atom_idx, 
                                                                    msms_bin=msms_bin,
                                                                    workdir=workdir,
                                                                    probe = probe)
                                                                           
        # Find the distance between every vertex in binding site surface and each atom in the ligand.
        kdt = KDTree(atomCoords)
        d, r = kdt.query(vertices1)
        assert(len(d) == len(vertices1))
        iface_v = np.where(d <= dist_threshold)[0]
        faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
        
        # Compute "charged" vertices
        if use_hbond:
            vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
        
        # For each surface residue, assign the hydrophobicity of its amino acid. 
        if use_hphob:
            vertex_hphobicity = computeHydrophobicity(names1) 
        
        vertices2 = vertices1
        faces2 = faces1
        # Fix the mesh.
        mesh = pymesh.form_mesh(vertices2, faces2)
        mesh = pymesh.submesh(mesh, faces_to_keep, 0)
        with io.capture_output() as captured:
            regular_mesh = fix_mesh(mesh, mesh_res)
    
    except:
        try:
            dist = [[distance.euclidean(ac, a.get_coord()) for ac in atomCoords] for a in atoms]
            atom_idx = np.argsort(np.min(dist, axis=1))[0]
            vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                        protonate=True, 
                                                                        one_cavity=atom_idx, 
                                                                        msms_bin=msms_bin,
                                                                        workdir=workdir,
                                                                        probe = probe)
                                                                           
            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
            
            # Compute "charged" vertices
            if use_hbond:
                vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
                
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if use_hphob:
                vertex_hphobicity = computeHydrophobicity(names1) 
            
            vertices2 = vertices1
            faces2 = faces1
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, mesh_res)
        except:
            vertices1, faces1, normals1, names1, areas1 = computeMSMS("%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold),  
                                                                        protonate=True, 
                                                                        one_cavity=None, 
                                                                        msms_bin=msms_bin,
                                                                        workdir=workdir,
                                                                        probe = probe)
            
            # Find the distance between every vertex in binding site surface and each atom in the ligand.
            kdt = KDTree(atomCoords)
            d, r = kdt.query(vertices1)
            assert(len(d) == len(vertices1))
            iface_v = np.where(d <= dist_threshold)[0]
            faces_to_keep = [idx for idx, face in enumerate(faces1) if all(v in iface_v  for v in face)] 
            
            # Compute "charged" vertices
            if use_hbond:
                vertex_hbond = computeCharges(prot_path.replace(".pdb",""), vertices1, names1)    
                
            # For each surface residue, assign the hydrophobicity of its amino acid. 
            if use_hphob:
                vertex_hphobicity = computeHydrophobicity(names1) 
            
            vertices2 = vertices1
            faces2 = faces1
            # Fix the mesh.
            mesh = pymesh.form_mesh(vertices2, faces2)
            mesh = pymesh.submesh(mesh, faces_to_keep, 0)
            with io.capture_output() as captured:
                regular_mesh = fix_mesh(mesh, mesh_res)

    ## resolve all degeneracies
    regular_mesh, info = pymesh.remove_degenerated_triangles(regular_mesh)

    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces, eps=epsilon)

    # Assign charges on new vertices based on charges of old vertices (nearest neighbor)
    if use_hbond:
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, feature_interpolation)

    if use_hphob:
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, feature_interpolation)

    if use_apbs:
        vertex_charges = computeAPBS(regular_mesh.vertices, "%s/%s_all_%s.pdb"%(workdir, protname, dist_threshold), 
                                    apbs_bin, pdb2pqr_bin, multivalue_bin, workdir)

    # Compute the principal curvature components for the shape index. 
    regular_mesh.add_attribute("vertex_mean_curvature")
    H = regular_mesh.get_attribute("vertex_mean_curvature")
    regular_mesh.add_attribute("vertex_gaussian_curvature")
    K = regular_mesh.get_attribute("vertex_gaussian_curvature")
    print('6')
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)

    print('==============save_ply==============')
    # Convert to ply and save.
    if out_name == None:
        print("%s/%s_all_%s.ply"%(outdir, protname, dist_threshold))
        save_ply("%s/%s_all_%s.ply"%(outdir, protname, dist_threshold), regular_mesh.vertices,
                regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,
                normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
                si=si)
    else:
        save_ply("%s/%s_all_%s.ply"%(outdir, out_name, dist_threshold), regular_mesh.vertices,
        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,
        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
        si=si)
    shutil.rmtree(workdir)

    
    