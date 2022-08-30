"""
fasta='step2/prot_sequences.fasta' # peptide_sequences.fasta
outpath='step2/ss/prot_ssp'
scratchtool="/home/ec2-user/SageMaker/efs/data/CAMP/tools/SCRATCH-1D_2.0/bin/run_scratch1d_predictors.sh --input_fasta $fasta --output_prefix $outpath "

fasta='step2/peptide_sequences.fasta' # peptide_sequences.fasta
outpath='step2/ss/peptide_ssp'
scratchtool="/home/ec2-user/SageMaker/efs/data/CAMP/tools/SCRATCH-1D_2.0/bin/run_scratch1d_predictors.sh --input_fasta $fasta --output_prefix $outpath "
"""
import pandas as pd

#Secondary Structure
seqtype = 'prot' # prot or peptide
output_ss_filename = 'step3/' + seqtype + '_seq_ss.txt'

# Generate secondary structure predictions first  
scratchtool="/home/ec2-user/SageMaker/efs/data/CAMP/tools/SCRATCH-1D_2.0/bin/run_scratch1d_predictors.sh "

querytype = 'prot' #prot or peptide
input_fasta = './step2/' + querytype + '_sequences.fasta'

outdir = os.path.abspath('./step2/ss/')
outpath = outdir + querytype + '_ssp'
process = subprocess.Popen([scratchtool + ' --input_fasta ' + input_fasta + ' --output_prefix ' + outpath])
stdout, stderr = (process.communicate())

# load predicted ss features for sequences in the dataset
def aa_ss_concat(aa,ss):
    if len(aa)!= len(ss):
        return 'string length error!'
    else:
        new_str = ''
        for i in range(len(aa)):
            concat_str = aa[i]+ss[i]+','
            new_str = new_str+concat_str
    final_str = new_str[:-1]
    return final_str


#df_org = pd.read_csv('./ss/seq_data.out.ss',sep='#',header = None) #the generated file by SCRATCH1D SSPro
df_org = pd.read_csv('./step2/ss/' + seqtype + '_ssp.ss3',sep='#',header = None) #the generated file by SCRATCH1D SSPro
df_org.columns = ['col_1']

# subset sequence dataframe and sse dataframe
df_seqid = df_org.iloc[::4, ] # .iloc[seq_idx]
df_seqid.columns = ['seq_id']
df_seqid.loc[:, 'seq_id'] = df_seqid['seq_id'].str.replace('>', '')
df_seq = df_org.iloc[1::4, ] # .iloc[seq_idx]
df_seq.columns = ['seq']
df_ss = df_org.iloc[2::4, ]
df_ss.columns = ['seq_ss']

df_seqid = df_seqid.reset_index(drop=True)
df_seq = df_seq.reset_index(drop=True)
df_ss = df_ss.reset_index(drop=True)

# join sequence & sse together
df_seq_ss = pd.merge(df_seqid, df_ss,left_index=True, right_index=True)

df_output_ss = pd.merge(df_seq_ss, df_seq,left_index=True, right_index=True)
df_output_ss['concat_seq'] = df_output_ss.apply(lambda x: aa_ss_concat(x['seq'],x['seq_ss']),axis=1)
df_output_ss.to_csv(output_ss_filename, encoding = 'utf-8', index = False, sep = '\t') # 'output_ss_filename' is the name of the output tsv you like
            
