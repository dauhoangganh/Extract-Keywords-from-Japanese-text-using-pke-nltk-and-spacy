#Extract keywords from 研究開発課題名, 当年度目的（一部【非公開】表示あり）
import argparse
import pandas as pd
import re
#from sklearn.feature_extraction.text import TfidfVectorizer
import pke
import nltk
from spacy.lang.ja import stop_words
from collections import defaultdict
from pke.base import  is_file_path
import os
import gzip
import math

pke.base.lang_stopwords['ja_ginza']= 'japanese'
stopwords=list(stop_words.STOP_WORDS)
nltk.corpus.stopwords.words_org= nltk.corpus.stopwords.words
nltk.corpus.stopwords.words= lambda lang: stopwords if lang == 'japanese' else nltk.corpus.stopwords.words_org(lang)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputCSV', required=True,help="name of input csv file")
    parser.add_argument('--OutputCSV',required=True, help='Name of output csv file')
    parser.add_argument('--targetFieldname',required=True, help='name of the keyword field')
    parser.add_argument('--InputDir',required=True, help='directory of input file')
    args = parser.parse_args()

# compute df counts 
def compute_document_frequency(input, output_file='df.tsv.gz',
                                language='ja_ginza',                # language of files
                                normalization=None,    # use porter stemmer
                                 encoding='utf8', delimiter='\t'):
    nb_documents = 0
    frequencies=defaultdict(int)
    # check if input is a dir or a file path
    if is_file_path(input) and input.endswith('.txt'):
        documents = open(input, 'r', encoding='utf8').read().splitlines()
    for docoument in documents:
        doc = pke.unsupervised.TfIdf()
        doc.load_document(input=docoument,
                          language=language,
                          normalization=normalization,
                          encoding=encoding)
        doc.candidate_selection(stoplist=stopwords)
        for lexical_form in doc.candidates:
            frequencies[lexical_form] += 1
        nb_documents += 1
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # dump the df container
    with gzip.open(output_file, 'wt', encoding='utf-8') as f:
        # add the number of documents as special token
        first_line = '--NB_DOC--' + delimiter + str(nb_documents)
        f.write(first_line + '\n')
        for ngram in frequencies:
            line = ngram + delimiter + str(frequencies[ngram])
            f.write(line + '\n')
file=pd.read_csv(args.InputCSV, index_col=None, skipinitialspace=True, skiprows=[1,2])
new_file=file[['研究開発課題名','当年度目的（一部【非公開】表示あり）']] #make new txt file of target columns for tfidf calculation
new_file.to_csv('targetcol.txt', sep=',', encoding='utf-8', header=False, quotechar='"', index=False)

compute_document_frequency(input=os.path.join(args.InputDir, 'targetcol.txt'), output_file='df.tsv.gz',
                                language='ja_ginza',                # language of files
                                normalization='lemmatization',    # use porter stemmer
                                 encoding='utf8', delimiter='\t')   #export df.tsv.gz file for df counts


def find_fre(row):
    key_fre_lst=[]
    targetcol=str(row['研究開発課題名'])+','+str(row['当年度目的（一部【非公開】表示あり）'])
    sakuingo=str(row['索引語（非公開）'])+str(row['索引語上位語（非公開）'])
    extractor=pke.unsupervised.TfIdf()
    extractor.load_document(input=targetcol, language='ja_ginza', normalization=None)
    extractor.candidate_selection(stoplist=stopwords)
    df=pke.load_document_frequency_file(input_file='df.tsv.gz')
    extractor.candidate_weighting(df=df)
    keyphrases=extractor.get_n_best(n=20)
    normalization=0
    for phrase, score in keyphrases:
        normalization += score*score
    for i in keyphrases:
        if i[0].replace(" ", "") not in sakuingo or i[0] not in sakuingo:
            r=re.findall(i[0].replace(" ", ""), targetcol)
            r2=re.findall(i[0], targetcol)
            S=str(i[0])+' / '+str(format(i[1]/math.sqrt(normalization),'.4f'))+' / '+str(len(r)+len(r2))
            key_fre_lst.append(S)
    return '|'.join(key_fre_lst)

new_file[args.targetFieldname]=new_file.apply(find_fre, axis=1)
file[args.targetFieldname]=new_file[args.targetFieldname]

#Add field_type and field_flag rows to dataframe
import csv
with open(args.InputCSV, 'r', encoding='utf8') as csvfile:
    file2=csv.reader(csvfile, skipinitialspace=True, escapechar='\\', quotechar='"')
    field_names=next(file2)
    lines=[]
    for i, line in enumerate(file2):
        if i==0:
            field_types=line
        elif i==1:
            field_flags=line
fn_ftype_dict={}
fn_fflag_dict={}
for fname, ftype in zip(field_names, field_types):
    fn_ftype_dict[fname]=ftype
for fname, fflag in zip(field_names, field_flags):
    fn_fflag_dict[fname]=fflag
fn_ftype_dict[args.targetFieldname]='string'
fn_fflag_dict[args.targetFieldname]='0'
file.loc[-1]=fn_fflag_dict
file.index = file.index + 1
file = file.sort_index()
file.loc[-1]=fn_ftype_dict
file.index = file.index + 1
file = file.sort_index()

#export dataframe to csv
file1=pd.DataFrame(file)
file1.to_csv(args.OutputCSV, sep=',', encoding='utf-8', header=True, quotechar='"', index=False)

