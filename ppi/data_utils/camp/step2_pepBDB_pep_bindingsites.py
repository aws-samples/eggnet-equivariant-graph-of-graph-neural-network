# -*- coding: utf-8 -*-
import pandas as pd

"""
BEFORE THIS STEP: (Jasleen added):

1a. Prepare pepbdb-2020/pepbdb directory from .tgz download at http://huanglab.phys.hust.edu.cn/pepbdb/db/download/  [DONE] - /home/ec2-user/SageMaker/efs/data/CAMP/paper/pepbdb

1b. Run crawl.py to generate crawl_results.csv [ONGOING]

2. Next, run query-mapping.py that uses inputs crawl_results.csv, and the peptide.pdb files in pepbdb-20200318/pepbdb/{pdbid}
- This script outputs peptide-mapping.txt, query_peptide_sequence_index.txt

"""
# Step 1:  According to the "PDB ID-Peptide Chain-Protein Chain" obtained in "step1_pdb_process.py" , retrieve the interacting information with following fields:
# ("Peptide ID","Interacting peptide residues","Peptide sequence","Interacting receptor residues","Receptor sequence(s)") a
# nd downloading the corresponding "peptide.pdb" files (please put under ./pepbdb-2020/pepbdb/$pdb_id$/peptide.pdb)

# Step 2: To map the peptide sequences from PepBDB to the peptide sequences from the peptide sequences from the RCSB PDB() generated in "step1_pdb_process.py").

# Generate query (PepBDB version) sequence file called "query_peptide.fasta" & target (RSCB PDB) fasta sequence files called "target_peptide.fasta" for peptides
# We use scripts under ./smith-waterman-src/ to align two versions of peptide sequences. The output is "alignment_result.txt"
#python query_mapping.py #to get peptide sequence vectors (the output is "peptide-mapping.txt ")
#python target_mapping.py #to get target sequence vector

# Step 3: Loading and mapping labels of binding residues for peptide sequences
# load peptide-protein pairs & pepBDB files (target : PDB fasta, query : pepBDB)
df_train = pd.read_csv('step1/pdb_chain_uniprot-processed.tsv', header=0, sep='\t', comment='#') # The output of "step1_pdb_process.py"

df_zy_pep = pd.read_csv('step2/peptide-mapping.txt',header=None,sep='\t')
df_zy_pep.columns= ['bdb_id','bdb_pep_seq','pep_binding_vec']
df_zy_pep['pdb_id'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[0])
df_zy_pep['pep_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[1].lower())
df_zy_pep['prot_chain'] = df_zy_pep.bdb_id.apply(lambda x: x.split('_')[2].upper())
df_zy_pep.drop_duplicates(['bdb_id'],inplace=True)

# Since we did not run uniprot based MHC filtering step in Step 1, 
# We don't have the protein chain column defined. Instead, we do that 
# modification here (same as 'plip_prot_chain' in skipped step). 
df_train['prot_chain'] = df_train.predicted_chain.apply(lambda x:\
                                                                  x.upper())
df_join = pd.merge(df_train, df_zy_pep, how='left', left_on=['pdb_id','pep_chain','prot_chain'],right_on=['pdb_id','pep_chain','prot_chain'])
#df_v1 = df_join[['pdb_id','pep_chain','prot_chain','pep_seq','SP_PRIMARY','prot_seq','Protein_families','pep_binding_vec']]
df_v1 = df_join[['pdb_id','pep_chain','prot_chain','pep_seq','SP_PRIMARY','prot_seq','pep_binding_vec']]
print(df_v1.shape)

# impute records that don't have bs information with -99999
def extract_inter_idx(pep_seq,binding_vec):
    if binding_vec==binding_vec:
        if len(binding_vec) != len(pep_seq):
            print('Error length')
            return '-99999'
        else:
            binding_lst = []
            for idx in range(len(binding_vec)):
                if binding_vec[idx]=='1':
                    binding_lst.append(idx)
            binding_str = ','.join(str(e) for e in binding_lst)
            return binding_str
    else:
        return '-99999'
    
df_v1['binding_idx'] = df_v1.apply(lambda x: extract_inter_idx(x.pep_seq,x.pep_binding_vec),axis=1)
#df_part_pair = df_part_all[['pep_seq','prot_seq','binding_idx']]
df_part_pair = df_v1[['pep_seq','prot_seq','binding_idx']]
df_pos_bs = pd.merge(df_v1,df_part_pair,how='left',left_on=['pep_seq','prot_seq'],right_on=['pep_seq','prot_seq']).drop_duplicates().reset_index()
df_pos_bs.to_csv('step2/pdb_pairs_bindingsites', encoding = 'utf-8', index = False, sep = ',')

ofile = open('step2/peptide_sequences.fasta', 'w')
for i in range(df_pos_bs.shape[0]):
    ofile.write('>' + df_pos_bs.loc[i, 'pdb_id'] + '_' + df_pos_bs.loc[i, 'pep_chain'] + '\n' + df_pos_bs.loc[i, 'pep_seq'] + '\n')
ofile.close()

ofile = open('step2/prot_sequences.fasta', 'w')
for i in range(df_pos_bs.shape[0]):
    ofile.write('>' + df_pos_bs.loc[i, 'pdb_id'] + '_' + df_pos_bs.loc[i, 'prot_chain'] + '\n' + df_pos_bs.loc[i, 'prot_seq'] + '\n')
ofile.close()

