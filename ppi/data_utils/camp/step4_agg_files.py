"""
This script combines all the necessary files and organizes them as needed by preprocess_features.py
"""
import pandas as pd
ss_prot = './step3/prot_seq_ss.txt'
ss_peptide = './step3/peptide_seq_ss.txt'

# Format the peptide-protein data like 
# seq, pep, label, pep_ss, seq_ss
prot_df = pd.read_csv(ss_prot, sep='\t')
pep_df = pd.read_csv(ss_peptide, sep='\t')
prot_df.columns = [x + '_prot' for x in prot_df.columns.tolist()]
pep_df.columns = [x + '_pep' for x in pep_df.columns.tolist()]

pairs_df = pd.read_csv("step2/pdb_pairs_bindingsites", sep=",")
mappings = pairs_df[['pdb_id', 'pep_chain', 'prot_chain']]
mappings['seq_id_pep'] = mappings['pdb_id'] + '_' + mappings['pep_chain']
mappings['seq_id_prot'] = mappings['pdb_id'] + '_' + mappings['prot_chain']

merged_df = pd.concat([pd.concat([prot_df, mappings], axis=1, join='inner'), pep_df], axis=1, join='inner')
merged_df['label'] = merged_df['pdb_id'] + "_" + merged_df['pep_chain'] + "_" + merged_df['prot_chain']
#out_df = merged_df[['seq_prot', 'seq_pep', 'label', 'seq_ss_pep', 'seq_ss_prot']]
out_df = merged_df[['seq_prot', 'seq_pep', 'label', 'concat_seq_pep', 'concat_seq_prot']]
out_df.to_csv('test_filename', encoding = 'utf-8', index = False, sep = '\t') 

out_df = merged_df[['seq_prot', 'seq_pep', 'concat_seq_pep', 'concat_seq_prot']]
out_df.to_csv('test_data.tsv', encoding = 'utf-8', index = False, sep = '\t') 

## Also copy over the pssm dict (protein) and the intrinsic disorder dicst (protein and peptide)
import shutil
shutil.copy2('./step3/pssm_prot.pkl', './dense_feature_dict/Protein_pssm_dict')
shutil.copy2('./step3/intrinsic_dict_prot.pkl', './dense_feature_dict/Protein_Intrinsic_dict') 
shutil.copy2('./step3/intrinsic_dict_peptide.pkl', './dense_feature_dict/Peptide_Intrinsic_dict_v3')
