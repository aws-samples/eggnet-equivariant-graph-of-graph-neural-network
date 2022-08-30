"""
This script takes in the sequence fasta files generated from Step2 and processes each sequence in each fasta file through the iupred tool. Note that the input files have to be of naming convention step2/[prot,peptide]_sequences.fasta, and that the results will be stored in the subfolder step3, in the current working directory.  
"""

import pandas as pd
import numpy as np
import os
import subprocess
import pickle

iupredtool="python3 /home/ec2-user/SageMaker/efs/data/CAMP/tools/iupred2a/iupred2a.py "
querytype = 'prot' # prot or peptide

fasta_file = './step2/' + querytype + '_sequences.fasta'

outdir = os.path.abspath('./step3/')
output_intrinsic_dict = outdir + '/intrinsic_dict_' + querytype + '.pkl'

if not os.path.exists(os.path.abspath(outdir + '/tmp/')):
    os.makedirs(os.path.abspath(outdir + '/tmp'))  

def get_iupred_rawscores(fasta_file, outdir, ind, iupredtool=iupredtool):
    ## This command runs iupred2a for each sequence 
    ## (each sequence must be a separate fasta file, so we create temp subset files from main fasta)
    ## This is because iupred2a parses all sequences in a fasta as one long sequence -_- 
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
    # Then run iupred for each sequence
    longseq_pred = {}
    for ix in range(len(id_list)):
        # Write out each sequence as a fasta file
        outfile_seq = os.path.abspath(outdir + '/tmp/' + id_list[ix].strip('>') + '.fasta')
        with open(outfile_seq, 'w') as outfile:
            for elem in [id_list[ix], seq_list[ix]]:
                outfile.write(elem + '\n')
        # Run that sequence through Intrinsic Disorder prediction (long, short)
        process = subprocess.Popen([iupredtool + outfile_seq + ' ' + ind], stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
        process.wait()
        # Store the results
        tmpscores = []
        for line in process.stdout.readlines():
            line_list = line.decode("utf-8").strip()
            if (len(line_list)>0 and line_list[0]!='#'):
                tmpscores.append(line_list)
        longseq_pred[id_list[ix]] = tmpscores
    return longseq_pred

# Load Intrinsic disorder
# dict : {sequence: Intrinsic Disorder Matrix}
# Intrinsic Disorder Matrix : (sequence length ,3) , last dimension :(long , short, ANCHOR score)
def load_fasta(fasta_filename):
    raw_fasta_list = []
    with open(fasta_filename,'r') as f:
        for line in f.readlines():
            line_list = line.strip()
            raw_fasta_list.append(line_list)
    fasta_id_list = [x for x in raw_fasta_list if x[0]=='>']
    fasta_sequence_list = [x for x in raw_fasta_list if x[0]!='>']
    fasta_seq_len_list = [len(x) for x in fasta_sequence_list]
    print(len(fasta_id_list),len(fasta_sequence_list),len(fasta_seq_len_list))
    fasta_dict = {}
    for i in range(len(fasta_id_list)):
        fasta_dict[fasta_id_list[i]]=(fasta_sequence_list[i],fasta_seq_len_list[i])
    return fasta_dict

def extract_intrinsic_disorder(fasta_filename, ind, outdir=outdir, iupredtool=iupredtool):
    fasta_dict = load_fasta(fasta_filename)
    raw_result_dict = get_iupred_rawscores(fasta_filename, outdir, ind, iupredtool=iupredtool)
    intrinsic_id_list = list(raw_result_dict.keys())
    raw_score_dict = {}
    for idx in range(len(intrinsic_id_list)):
        prot_id = intrinsic_id_list[idx]
        seq_len = fasta_dict[prot_id][1]
        individual_score_list = [x.split('\t') for x in raw_result_dict[prot_id]]
        individual_score_list = [x[2:] for x in individual_score_list]
        individual_score_array = np.array(individual_score_list,dtype='float')
        raw_score_dict[prot_id] = individual_score_array
    print(len(fasta_dict.keys()),len(raw_score_dict.keys()))
    return fasta_dict, raw_score_dict

# long & short
# the input fasta file used in IUPred2A
fasta_dict_long, raw_score_dict_long = extract_intrinsic_disorder(fasta_filename=fasta_file, ind='long') 
fasta_dict_short, raw_score_dict_short = extract_intrinsic_disorder(fasta_filename=fasta_file, ind='short')

Intrinsic_score_long = {}
for key in fasta_dict_long.keys():
    sequence = fasta_dict_long[key][0]
    seq_len = fasta_dict_long[key][1]
    Intrinsic = raw_score_dict_long[key]
    if Intrinsic.shape[0]!= seq_len:
        print('Error!')
    Intrinsic_score_long[sequence] = Intrinsic

    
Intrinsic_score_short = {}
for key in fasta_dict_short.keys():
    sequence = fasta_dict_short[key][0]
    seq_len = fasta_dict_short[key][1]
    Intrinsic = raw_score_dict_short[key]
    if Intrinsic.shape[0]!= seq_len:
        print('Error!')
    Intrinsic_score_short[sequence] = Intrinsic

Intrinsic_score = {}
for seq in Intrinsic_score_short.keys():
    long_Intrinsic = Intrinsic_score_long[seq][:,0]
    short_Intrinsic = Intrinsic_score_short[seq]
    concat_Intrinsic = np.column_stack((long_Intrinsic,short_Intrinsic))
    Intrinsic_score[seq] = np.column_stack((long_Intrinsic,short_Intrinsic))


with open(output_intrisic_dict,'wb') as f: # 'output_intrisic_dict' is the name of the output dict you like
    pickle.dump(Intrinsic_score,f)
            
