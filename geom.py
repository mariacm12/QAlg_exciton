#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:30:47 2020

@author: mariacm
"""

import numpy as np
from numpy import linalg as LA
from math import sqrt, pi, cos, sin
from scipy import linalg as sLA

#MD analysis tools
import MDAnalysis as mda
import MDAnalysis.analysis.distances as MDdist

ams_au = 0.52917721092
cm1toHartree = 4.5563e-6
cm1_to_ev = 0.00012 #1 cm^-1


class Geom:
    def __init__(self, filepdb1, filepdb2,
                 pcharge1=None, pcharge2=None,
                 TEnergy1=0, TEnergy2=0,
                 tdm1=0, tdm2=0
                 ):  
        
        # Set initial coordinates of dimer
        
        u1 = mda.Universe(filepdb1)
        u2 = mda.Universe(filepdb2)

        self.MonA = u1.select_atoms("all")
        self.MonB = u2.select_atoms("all")
        
        self.xyzA = self.MonA.positions
        self.xyzB = self.MonB.positions
        
        self.cmass_A = self.MonA.center_of_mass()
        self.cmass_B = self.MonB.center_of_mass()
        
        # Rotation operators 
        self.Rz = lambda th: np.array([[cos(th),-sin(th),0],
                                  [sin(th),cos(th),0],
                                  [0,0,1]])
        self.Ry = lambda th: np.array([[cos(th),0,sin(th)],
                                  [0,1,0],
                                  [-sin(th),0,cos(th)]])
        self.Rx = lambda th: np.array([[1,0,0],
                                  [0,cos(th),-sin(th)],
                                  [0,sin(th),cos(th)]])
        
        # Partial charges calculated for the monomers (remain constant)
        self.chrgAl = pcharge1
        self.chrgBl = pcharge2
                                     
        # Transition energies
        self.TEnergy1 = TEnergy1
        self.TEnergy2 = TEnergy2
        
        #Transition dipole moment
        self.TDM1 = tdm1
        self.TDM2 = tdm2
        
        self.th0 = [0,0,0] #initial angle for the fixed molecule


    def max_rij(self):
        '''
        Max distance after which coupling is negligable
    
        Parameters
        ----------
        miua : ndarray
        miub : ndaray
    
        Returns
        -------
        rmax in a.u..
    
        '''
        norm1, norm2 = LA.norm(self.TDM1), LA.norm(self.TDM2)
        return (norm1*norm2*(3+1)/0.01)**(1/3) #in atomic units

    def atom_dist(self, a1, a2, coord1, coord2):
        '''
        Distance between two atoms
    
        Parameters
        ----------
        a1 : int
            index of atom 1.
        a2 : int
            index of atom 2.
        coord1 : ndarray
            array listing molecule's #1 coordinaties.
        coord2 : TYPE
            array listing molecule's #2 coordinaties.
    
        Returns
        -------
        dist : float
            in Amstrongs
        '''
        dist =LA.norm(coord2[a2] - coord1[a1])
        return dist

    def multipole_coup(self, pos1, pos2, at1, at2, atnH1, atnH2, chg1, chg2):
        '''
        Calculates multiple coupling from inter-atomic distance and atomic excited
            state partial charges
    
        Parameters
        ----------
        pos1 : ndarray
            cartesian coord of atoms in molecule 1.
        pos2 : ndarray
            cartesian coord of atoms in molecule 2.
        at1, at2 : ndarray
            list of indexes of atoms in molecule 1 and 2.
        atnH1, atnH2 : ndarray
            list of indexes of non-H atoms in molecule 1 and 2.
    
        Returns
        -------
        Vij : float
    
        '''
        
        at1 = at1-at1[0]
        at2 = at2-at2[0]
        
        distsum = np.array([[self.atom_dist(a1,a2,pos1,pos2) for a2 in at2-1] for a1 in at1-1]) #array (natoms1 x natoms2)
        
        #distance between non-H atoms only
        distsum_noH = np.array([[self.atom_dist(a1,a2,pos1,pos2) for a2 in atnH2-1] for a1 in atnH1-1]) #array (natoms1 x natoms2)
    
        # To avoid unphysically close constructs. No more than tol non-H atoms can be closer than 2A.
        #    No atoms can be closer than 1A
        if np.count_nonzero((np.abs(distsum_noH) < 2.0)) > 5 or np.any((np.abs(distsum_noH) < 1.0)):
            #print('tot = ', distsum[np.abs(distsum) < 2.0],', from:', len(at1),len(at2))
            Vij = 9999999999999999
            # print(np.count_nonzero((np.abs(distsum_noH) < 2.0)))
        else:
            Vij = np.sum( np.multiply(np.outer(chg1,chg2),1/distsum) ) #SUM_{f,g}[ (qf qg)/|rf-rg| ]
    
        return Vij

    def mtp_transform(self, cms, angles, GA=True):
        '''
        Calculates multipole-coupling from GA results 
    
        Parameters
        ----------
    
        Returns
        -------
        (V01,V02,V02,V12,V13,V23,tooclose)
    
        '''
        
        mol1 = self.MonA 
        mol2 = self.MonB
        
        cm0,cm1,cm2,cm3 = cms
        rang0,rang1,rang2,rang3 = angles
        
        
        atA = mol1.atoms.ids
        atB = mol2.atoms.ids
        
        #list of non-H atoms
        namesA = mol1.atoms.names
        nonHsA = np.invert(np.char.startswith(namesA.astype(str),'H'))
        atnonH_A = atA[nonHsA]
        namesB = mol2.atoms.names
        nonHsB = np.invert(np.char.startswith(namesB.astype(str),'H'))
        atnonH_B = atB[nonHsB]
        
        #moving mola and molb frame to origin    
        xyz_a = mol1.positions - mol1.center_of_mass()
        xyz_b = mol2.positions - mol2.center_of_mass()
        
        #Traslation to each dye's position
        xyz1, xyz2, xyz3, xyz4 = xyz_a, xyz_b, xyz_a, xyz_b 
    
        xyz0 = xyz1 + cm0
        xyz1 = xyz2 + cm1
        xyz2 = xyz3 + cm2
        xyz3 = xyz4 + cm3
    
    
        #Rotates it by the given angles
        xyzD1 = np.dot(np.dot(self.Rx(rang0[0]),self.Ry(rang0[1])),np.dot(self.Rz(rang0[2]),xyz0.T)).T
        xyzD2 = np.dot(np.dot(self.Rx(rang1[0]),self.Ry(rang1[1])),np.dot(self.Rz(rang1[2]),xyz1.T)).T
        xyzD3 = np.dot(np.dot(self.Rx(rang2[0]),self.Ry(rang2[1])),np.dot(self.Rz(rang2[2]),xyz2.T)).T
        xyzD4 = np.dot(np.dot(self.Rx(rang3[0]),self.Ry(rang3[1])),np.dot(self.Rz(rang3[2]),xyz3.T)).T
        
        at1, at2, at3, at4 = atA, atB, atA, atB
        atH1, atH2, atH3, atH4 = atnonH_A, atnonH_B, atnonH_A, atnonH_B
        chg1, chg2, chg3, chg4 = self.chrgAl, self.chrgBl, self.chrgAl, self.chrgBl
    
        tooclose = False
        
        if not GA:  
            xyzD3 += [30,30,30]
            xyzD4 += [30,30,30]
          
        V12 = self.multipole_coup(xyzD1,xyzD2,at1,at2,atH1,atH2,chg1,chg2)/cm1toHartree
        V23 = self.multipole_coup(xyzD2,xyzD3,at2,at3,atH2,atH3,chg2,chg3)/cm1toHartree 
        V24 = self.multipole_coup(xyzD2,xyzD4,at2,at4,atH2,atH4,chg2,chg4)/cm1toHartree 
        V13 = self.multipole_coup(xyzD1,xyzD3,at1,at3,atH1,atH3,chg1,chg3)/cm1toHartree 
        V14 = self.multipole_coup(xyzD1,xyzD4,at1,at4,atH1,atH4,chg1,chg4)/cm1toHartree 
        V34 = self.multipole_coup(xyzD3,xyzD4,at3,at4,atH3,atH4,chg3,chg4)/cm1toHartree 
        
        
        # Condition set in multipole_coup 
        if any(np.array([V12,V13,V14,V23,V24,V34]) > 100000):
            tooclose = True
        #else: print("there's hope!!")
            
        return V12,V13,V14,V23,V24,V34, tooclose
    
    
    def solve_mtp(self, rab, ham, npoints, prec_re,prec_im=0.1):
        """
        Find optimal geometry by looping throguh the distance and angle parameters
            (alternative to GA when we only have a 2-molecule system)

        Parameters
        ----------
        rab : float
            Distance between molecules in z axis
        ham : np array
            2x2 Hamiltonian.
        npoints : int
        prec_re : Desired precision for the real component
        prec_im : Desired precision for the imaginary component

        Returns
        -------
        list with elements [(rotx,roty,rotz),dr].
        The optimal set of parameters.

        """
        
        mol1 = self.MonA 
        mol2 = self.MonB
        
        atA = mol1.atoms.ids
        atB = mol2.atoms.ids
        #list of non-H atoms
        namesA = mol1.atoms.names
        nonHsA = np.invert(np.char.startswith(namesA.astype(str),'H'))
        atnonH_A = atA[nonHsA]
        namesB = mol2.atoms.names
        nonHsB = np.invert(np.char.startswith(namesB.astype(str),'H'))
        atnonH_B = atB[nonHsB]
        
        atHA, atHB = atnonH_A, atnonH_B
        
        #moving mola and molb frame to origin    
        xyz_a = mol1.positions - mol1.center_of_mass()
        xyz_b = mol2.positions - mol2.center_of_mass()
        
        ucoord = np.linspace(2.5, 5, int(npoints/2)+1)
        rotang = np.linspace(-pi, pi, npoints+1)
    
        dz = rab
        dE = abs(self.TEnergy1 - self.TEnergy2)
    
        i_12 = np.sort(np.unique(ham.diagonal(),return_index=True)[1])
        
        def Vc(xyz2):
            xyz1 = xyz_a
            return self.multipole_coup(xyz1,xyz2,atA,atB,atHA,atHB)/cm1toHartree
    
        geom_set = []
        crit1 = prec_re
        crit2 = prec_im
        count = 0
        roty = 0
        
        #for rotx in rotang:
        for rotx in rotang:
            for rotz in rotang:
                for dx in ucoord:   
                    for dy in ucoord:
                        if count>=5:
                            break                   
                        ham12 = ham[i_12[0]][i_12[1]]                    
                        dr = np.array([dx,dy,dz])
            
                        xyzb = xyz_b + dr
                        xyzb = np.dot(np.dot(self.Rx(rotx),self.Ry(roty)),np.dot(self.Rz(rotz),xyzb.T)).T                           
                        Vij = Vc(xyzb)
                        
                        too_close = True if Vij > 10000 else False  
                        if not too_close: print(ham12*dE,Vij)
                        print(dE*ham12, i_12)
                        
                        if self.is_close(ham12,Vij/dE,crit1,crit2) and count<8 and not too_close:
                            print(too_close,Vij)
                            print(count,'*******rot=',[rotx,roty,rotz],', r=',[dx,dy,dz],ham12*dE)
                            geom_set.append([(rotx,roty,rotz),dr])      
                            count+=1                   
    
        return geom_set
        
    def make_pdb_GA(self, res_nameA, res_nameB):
        '''
        Makes transformed molecule object from GA optimization (for pdb saving)
    
        Parameters
        ----------
        res_nameA : string
            Name of the residue for saving molecule 1 in pdb.
        res_nameB : string
            Name of the residue for saving molecule 2 in pdb.
    
        Returns
        -------
        mol_new : MDAnalisys.AtomGroup
            Transformed molecule object.
    
        '''
        mol1 = self.MonA
        mol2 = self.MonB
        
        cm0,cm1,cm2,cm3 = self.cms
        rang0,rang1,rang2,rang3 = self.rot_angles
        
        #moving mola and molb frame to origin    
        xyz_a = mol1.positions - mol1.center_of_mass()
        xyz_b = mol2.positions - mol2.center_of_mass()
    
        #Traslation to each dye's position
        xyz0 = xyz_a + cm0
        xyz1 = xyz_b + cm1
        xyz2 = xyz_a + cm2
        xyz3 = xyz_b + cm3
    
        #Rotates it by the given angles
        xyzD1 = np.dot(np.dot(self.Rx(rang0[0]),self.Ry(rang0[1])),np.dot(self.Rz(rang0[2]),xyz0.T)).T
        xyzD2 = np.dot(np.dot(self.Rx(rang1[0]),self.Ry(rang1[1])),np.dot(self.Rz(rang1[2]),xyz1.T)).T
        xyzD3 = np.dot(np.dot(self.Rx(rang2[0]),self.Ry(rang2[1])),np.dot(self.Rz(rang2[2]),xyz2.T)).T
        xyzD4 = np.dot(np.dot(self.Rx(rang3[0]),self.Ry(rang3[1])),np.dot(self.Rz(rang3[2]),xyz3.T)).T
        
        #Creating new molecules
        nat1, nat2 = mol1.n_atoms, mol2.n_atoms
        resids = np.concatenate((np.full((nat1,1), 0),np.full((nat2,1), 1),np.full((nat1,1), 2),np.full((nat2,1), 3)),axis=0).T
        resnames = [res_nameA+'A']+[res_nameB+'A']+[res_nameA+'B']+[res_nameB+'B']
        atnames = mol1.atoms.names.tolist()+mol2.atoms.names.tolist()+mol1.atoms.names.tolist()+mol2.atoms.names.tolist()
        attypes = mol1.atoms.types.tolist()+mol2.atoms.types.tolist()+mol1.atoms.types.tolist()+mol2.atoms.types.tolist()
    
        mol_new = mda.Universe.empty(2*nat1 + 2*nat2, 4, atom_resindex=resids.tolist()[0],trajectory=True)
        mol_new.add_TopologyAttr('name', atnames)
        mol_new.add_TopologyAttr('type', attypes)
        mol_new.add_TopologyAttr('resname', resnames)
        mol_new.add_TopologyAttr('resid', list(range(0, 4)))
        
        mol_new.atoms.positions = np.concatenate((xyzD1,xyzD2,xyzD3,xyzD4))
        
        d1_bd = [mol1.atoms.bonds.to_indices()[i] for i in range(len(mol1.atoms.bonds.to_indices()))]
        d2_bd = [mol2.atoms.bonds.to_indices()[i] + nat1 for i in range(len(mol2.atoms.bonds.to_indices()))]
        d3_bd = [mol1.atoms.bonds.to_indices()[i] + nat1 + nat2 for i in range(len(mol1.atoms.bonds.to_indices()))]
        d4_bd = [mol2.atoms.bonds.to_indices()[i] + 2*nat1 + nat2 for i in range(len(mol2.atoms.bonds.to_indices()))]
        bonds0 = np.concatenate((np.concatenate((d1_bd,d2_bd),axis=0),np.concatenate((d3_bd,d4_bd),axis=0)),axis=0)
        bonds = list(map(tuple, bonds0))
    
        
        mol_new.add_TopologyAttr('bonds', bonds)
    
        
        return mol_new

    def make_pdb(self, opt, res_nameA, res_nameB, order='12'):
        '''
        Makes transformed molecule object from non-GA (solve_dye) optimization
    
        Parameters
        ----------
        opt : ndarray
            result from non-GA optimization.
        res_nameA : string
            Name of the residue representing molecule 1 in pdb.
        res_nameB : string
            Name of the residue representing molecule 2 in pdb.
        order : string
            which dyes are equal ('12' for HI, '13' dor DJC)
            
    
        Returns
        -------
        mol_new : MD Analysis group.
    
        '''

        mol1i = self.MonA
        mol2i = self.MonB
    
        thx,thy,thz = opt[0]
        
        if order == '12':
            mol1, mol3 = mol1i
            mol2, mol4 = mol2i
            res_name1,res_name2,res_name3,res_name4 = res_nameA,res_nameA,res_nameB,res_nameB
            
        if order == '13':
            mol1, mol2 = mol1i
            mol3, mol4 = mol2i
            res_name1,res_name2,res_name3,res_name4 = res_nameA,res_nameA,res_nameB,res_nameB
                        
        #moving mola and molb frame to origin    
        xyz_a = mol1.positions - mol1.center_of_mass()
        xyz_b = mol2.positions - mol2.center_of_mass()
        xyz_c = mol3.positions - mol3.center_of_mass()
        xyz_d = mol4.positions - mol4.center_of_mass()
        
    
        #Traslation to each dye's position
        xyzD1 = xyz_a #+ cm0
        xyz2  = xyz_b + opt[1]#cm1
        xyz3  = xyz_c         #cm2
        xyz4  = xyz_d + opt[1]#cm3
        
        #Rotates it by the given angles
        thx2 = thx4 = thx
        thy2 = thy4 = thy
        thz2 = thz4 = thz
        thx3 = thy3 = thz3 = 0
        xyzD2 = np.dot(np.dot(self.Rx(thx2),self.Ry(thy2)),np.dot(self.Rz(thz2),xyz2.T)).T
        xyzD3 = np.dot(np.dot(self.Rx(thx3),self.Ry(thy3)),np.dot(self.Rz(thz3),xyz3.T)).T
        xyzD4 = np.dot(np.dot(self.Rx(thx4),self.Ry(thy4)),np.dot(self.Rz(thz4),xyz4.T)).T
        
        #Moving dyes 2 and 3 away
        xyzD3 += [30,30,30]
        xyzD4 += [30,30,30]
        
        #Creating new molecules
        nat1, nat2, nat3, nat4 = mol1.n_atoms, mol2.n_atoms, mol3.n_atoms, mol4.n_atoms
        resids = np.concatenate((np.full((nat1,1), 0),np.full((nat2,1), 1),np.full((nat3,1), 2),np.full((nat4,1), 3)),axis=0).T
        resnames = [res_name1+'A']+[res_name2+'B']+[res_name3+'C']+[res_name4+'D']
        atnames = mol1.atoms.names.tolist()+mol2.atoms.names.tolist()+mol3.atoms.names.tolist()+mol4.atoms.names.tolist()
        attypes = mol1.atoms.types.tolist()+mol2.atoms.types.tolist()+mol3.atoms.types.tolist()+mol4.atoms.types.tolist()
    
        print(res_nameB,nat1,nat2,nat3,nat4)
        mol_new = mda.Universe.empty(nat1+nat2+nat3+nat4, 4, atom_resindex=resids.tolist()[0],trajectory=True)
        mol_new.add_TopologyAttr('name', atnames)
        mol_new.add_TopologyAttr('type', attypes)
        mol_new.add_TopologyAttr('resname', resnames)
        mol_new.add_TopologyAttr('resid', list(range(0, 4)))
        
        mol_new.atoms.positions = np.concatenate((xyzD1,xyzD2,xyzD3,xyzD4))
        
        d1_bd = [mol1.atoms.bonds.to_indices()[i] for i in range(len(mol1.atoms.bonds.to_indices()))]
        d2_bd = [mol2.atoms.bonds.to_indices()[i] + nat1 for i in range(len(mol2.atoms.bonds.to_indices()))]
        d3_bd = [mol3.atoms.bonds.to_indices()[i] + nat1 + nat2 for i in range(len(mol3.atoms.bonds.to_indices()))]
        d4_bd = [mol4.atoms.bonds.to_indices()[i] + nat1 + nat2 +nat3 for i in range(len(mol4.atoms.bonds.to_indices()))]
        bonds0 = np.concatenate((np.concatenate((d1_bd,d2_bd),axis=0),np.concatenate((d3_bd,d4_bd),axis=0)),axis=0)
        bonds = list(map(tuple, bonds0))
    
        
        mol_new.add_TopologyAttr('bonds', bonds)
    
        
        return mol_new 


    # Genetic algorithm supporting functions
    
    def dens_calc_ham(self, opt_param, which):
        '''
        Calculate the new hamiltonian from optimized parameters
    
        Parameters
        ----------
        opt_param : ndarray
            Vij geometric parameters optimized with the Genetic Alg.
        which : str
            "HH" or "HI", which gate Hamiltonian to calculate
    
        Returns
        -------
        ham_sys : ndarray 
            New Hamiltonian.
        tau : float
            Transformation time.
    
        '''
        import math
        hb = 5308.8 # in cm-1 * fs
        reorder = 0
        
        
        en1 = en3 = self.TEnergy1
        en2 = en4 = self.TEnergy2
        
        cm0 = self.cmass_A
        cm1 = self.cmass_B
    
        opt_param = opt_param.to_numpy()
        #Optimized centers of mass for molecules 2 and 3
        dcm1, dcm2, dcm3 = np.split(opt_param[:9],3)
        
        #Optimized "directional point" for all molecules
        th1, th2, th3 = np.split(opt_param[9:],3)
        
        if which == "HH":
            cm1 = dcm1
            cm2 = dcm2
            cm3 = dcm3
            GA = True    
            im_part = 0     
            reorder = 2
            
        elif which == "HI":
            cm1 = dcm1
            cm2 = dcm2
            cm3 = dcm3
            reorder = 1
            GA = False
            im_part = 0
        
        cms = (cm0,cm1,cm2,cm3)
        angles = (self.th0,th1,th2,th3)
        
        V01,V02,V03,V12,V13,V23,tooclose = self.mtp_transform(cms, angles, GA=GA)
    
        V01 += im_part
        V03 += im_part
        V12 += im_part
        V23 += im_part
        
        if not tooclose: 
            if reorder==1:
                V02,V01,V03,V13,V12,V23 = V01,V02,V03,V12,V13,V23
                en1, en2, en3, en4 = en1, en3, en2, en4
            elif reorder==2:
                V01,V03,V02,V13,V12,V23 = V01,V02,V03,V12,V13,V23
                en1, en2, en3, en4 = en1, en2, en4, en3           
                
            V10, V20, V30, V21, V31, V32 = np.conj(V01), np.conj(V02), np.conj(V03), np.conj(V12), np.conj(V13), np.conj(V23)
            
            #system hamiltonian
            ham_sys = np.array([[en1,V01,V02,V03],
                                [V10,en2,V12,V13],
                                [V20,V21,en3,V23],
                                [V30,V31,V32,en4]])
            tau = 1/(en1-en2)*(2)*(math.pi*hb/4) 
            
            if which=='HI':
                tau = np.abs(math.pi*hb/4 * sqrt(2)/V02)
            
        else: # The dyes are too close to each other!
            ham_sys = np.identity(4)
            tau = 99999
    
        return ham_sys,tau
    
    def ham_fitness(self, opt_param, which): 
        """
        

        Parameters
        ----------
        opt_param : ndarray
            Vij geometric parameters optimized with the Genetic Alg.
        which : str
            "HH" or "HI", which gate Hamiltonian to calculate

        Returns
        -------
        ham_sys: Hamiltonian from GA parameters
        fitness: Hamiltonian matrix fitness
        tau_new: New computation time from GA parameters

        """
        
        def Had_sol(e1, e2, div, c=2):
            deltaE = abs(e1 - e2)
            Vij = deltaE/c    
            tau=abs(div/Vij)
    
            return Vij, abs(tau)
        
        hb = 5308.8 # in cm-1 * fs
      
        
        en1 = en3 = self.TEnergy1
        en2 = en4 = self.TEnergy2        
        
        div = pi*hb/4    
        Vij, tau = Had_sol(en1,en2,div,c = 2) 
        V01, V02, V03, V12, V13, V23 = Vij,Vij,Vij,Vij,-Vij,-Vij
        
        pop = opt_param
            
        if which=='HI' or which =='DJC':
            thx,thy,thz = opt_param[0]
    
            dcm2,dcm3,dcm4 = opt_param[1],[0,0,0],opt_param[1]
    
            dth2 = [thx,thy,thz]
            dth3 = [0,0,0]
            dth4 = [thx,thy,thz]    
            
            pop0 = np.append(dcm2, [dcm3, dcm4])
            pop = np.append(pop0,[dth2,dth3,dth4])
            
            en1 = en2 = self.TEnergy1
            en3 = en4 = self.TEnergy2
            div = pi*hb/4
            Vij, tau = Had_sol(en1,en3,div,k=sqrt(2),c = 2)   
            V01=V02=V03=V12=V13=V23 = 0
            
            if which == 'HI':
                V02 = V13 = Vij
            else:
                V01 = V23 = Vij
                en2, en3 = en3, en2
    
        V10, V20, V30, V21, V31, V32 = V01, V02, V03, V12, V13, V23
        ham_orig = np.array([[en1,V01,V02,V03],
                            [V10,en2,V12,V13],
                            [V20,V21,en3,V23],
                            [V30,V31,V32,en4]])  
        
        # Calculating the fitness value of each solution in the current population.
        ham_sys, tau_new = self.dens_calc_ham(pop,which)
    
        # print('old',ham_orig,'\n New',ham_sys)
    
        if tau == 99999:
            fitness = 0
            print("too close :(")
        else:
            diff = np.matrix(ham_sys-ham_orig)*cm1_to_ev
            fitness0 = 1-(1/2)*np.trace(sLA.sqrtm(np.dot(diff.getH(),diff)))
            fitness = fitness0 if fitness0>0 else 0
        return ham_sys, fitness, tau_new
 

    
    def is_close(test_val, val, crit_re = 0.01, crit_imag=0.1):
       '''
       Determines whether the two numbers are approx the same
       
       Parameters
       ----------
       test_val : float
         value #1.
       val : float
         value #2.
       crit_re : float, optional
         Criteria to compare the 2 numbers. The default is 0.01. (real part)
       crit_re : float, optional
         Criteria to compare the 2 numbers. The default is 0.1. (imaginary part)
       
       Returns
       -------
       bool
       True if values are close enough. Otherwise False.
       
       '''
       if (not isinstance(test_val, complex)) or abs(test_val.imag)<0.00001: #when number is real
       
           if abs(test_val - val) > crit_re:
               return False
           else: return True        
       else:
           if (abs(test_val.real - val.real) > crit_re) or (abs(test_val.imag - val.imag) > crit_imag): 
               return False
           else: 
               return True
