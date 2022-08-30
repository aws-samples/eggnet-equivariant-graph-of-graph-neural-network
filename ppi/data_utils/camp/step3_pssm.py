"""
This script will 
a) generate .fasta and .pssm files for each individual protein sequence. 
b) process the pssms into a dict with the sequence IDs. 

# May first need to run perl /home/ec2-user/SageMaker/efs/data/CAMP/tools/ncbi-blast-2.13.0+/bin/update_blastdb.pl swissprot.fa

This script will run the following psiblast command for each sequence, to generate the PSSM: 
$psiblast -db swissprot -query $outdir/tmp/1a61_i.fasta -num_iterations 3 -evalue 0.001 -out_ascii_pssm test.pssm
"""

import pandas as pd
import numpy as np
import os
import subprocess
import pickle

### Generate Protein PSSM Files  
psiblast="/home/ec2-user/SageMaker/efs/data/CAMP/tools/ncbi-blast-2.13.0+/bin/psiblast "
blastdb='export BLASTDB=/home/ec2-user/SageMaker/efs/data/CAMP/tools/ncbi-blast-2.13.0+/bin/; '
psiblast_opts = '-db ' + swissprotdb + ' -num_iterations 3 -evalue 0.001 '
swissprotdb='swissprot'

querytype = 'prot' # prot or peptide

fasta_file = './step2/' + querytype + '_sequences.fasta'

outdir = os.path.abspath('./step3/')
output_pssm_dict = outdir + 'pssm_' + querytype + '.pkl'

if not os.path.exists(os.path.abspath(outdir + '/tmp/')):
    os.makedirs(os.path.abspath(outdir + '/tmp'))  

def get_pssm(fasta_file, outdir, psiblast=psiblast, psiblast_opts=psiblast_opts):
    ## This command runs psiblast for each sequence 
    ## (each sequence must be a separate fasta file, so we create temp subset files from main fasta)
    # First we get the input ids and sequences
    with open(fasta_file, 'r') as f:
        id_list = []
        seq_list = []
        for line in f.readlines():
            line = line.strip()
            if line.startswith('>'):
                id_list.append(line)
            else:
                if(len(line) > 0):
                    seq_list.append(line)
    # Then run psiblast for each sequence
    longseq_pred = {}
    for ix in range(len(id_list)):
        # Write out each sequence as a fasta file
        outfile_seq = os.path.abspath(outdir + '/tmp/' + id_list[ix].strip('>') + '.fasta')
        outfile_pssm = os.path.abspath(outdir + '/tmp/' + id_list[ix].strip('>') + '.pssm')
        if not os.path.exists(outfile_pssm):
            with open(outfile_seq, 'w') as outfile:
                for elem in [id_list[ix], seq_list[ix]]:
                    outfile.write(elem + '\n')
            # Run that sequence through psiblast prediction (long, short)
            process = subprocess.Popen([
                blastdb + 
                psiblast + psiblast_opts + '-query ' + outfile_seq + ' -out_ascii_pssm ' + outfile_pssm], stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
            stdout, stderr = (process.communicate())
    return id_list

proteins_list = get_pssm(fasta_file = fasta_file, outdir = outdir)

### Load Protein PSSM Files (first change the value of protein_number)
# prot_pssm_dict : key is protein sequence, value is protein PSSM Matrix
prot_pssm_dict_all={}
prot_pssm_dict={}
protein_num = len(proteins_list) ### NEED TO BE CHANGED TO the total number of protein sequences
inputs_dir = os.path.abspath(outdir + "tmp/")
for protid in proteins_list:
    filename_pssm = protid.strip('>') + '.pssm' # need to name each individual fasta and pssm file with the same prefix
    filename_fasta = protid.strip('>') + '.fasta'
    prot_key = protid.strip('>') # 'new_prot_'+str(i)
    pssm_line_list= []
    
    with open(inputs_dir+'/'+filename_fasta,'r') as f: # directory to store fasta files (single file of each protein)
        for line in f.readlines():
            prot_seq = line.strip()
    
    with open(inputs_dir+'/'+filename_pssm,'r') as f:  # directory to store pssm files (single file of each protein)
        for line in f.readlines()[3:-6]:
            line_list = line.strip().split(' ')
            line_list = [x for x in line_list if x!=''][2:22]
            line_list = [int(x) for x in line_list]
            if len(line_list)!=20:
                print('Error line:')
                print(line_list)
            pssm_line_list.append(line_list)
        pssm_array = np.array(pssm_line_list)
        if pssm_array.shape[1]!=20:
            print('Error!')
            print(filename_pssm)
        else:
            prot_pssm_dict_all[prot_key] = (prot_seq,pssm_array)
            prot_pssm_dict[prot_seq]=pssm_array

with open(output_pssm_dict,'wb') as f:  # 'output_pssm_dict' is the name of the output dict you like
    pickle.dump(prot_pssm_dict,f)
