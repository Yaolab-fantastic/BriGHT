import argparse
import numpy as np
import sys
import pandas as pd

def check_file(expression):
    checkset = set(["", "NA", "Na", "na", "nan", "null"])
    for c in checkset:
        loc = np.where(expression == c)
        if loc[0].size:
            expression[loc] = "0"
            print(f"Notice! There is {c} in the 'gene expression matrix' file and it will be assigned to 0.")
    return expression

def read_file(file_path):
    data = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                item = line.strip('\n').split('\t')[0]
                data.add(item)
    except FileNotFoundError:
        print(f"Error! File not found: {file_path}")
        sys.exit()
    return data

parser = argparse.ArgumentParser(description="Manual")
parser.add_argument("-f", type=str, default="path/to/input_expression.txt",
                    help="A path to 'gene expression matrix' file")
parser.add_argument("-w", type=str, default="path/to/sample_weight.txt",
                    help="A path to 'sample weight' file")
parser.add_argument("-p", type=str, default="path/to/samples_of_interest.txt",
                    help="A path to 'samples of interest' file")
parser.add_argument("-g", type=str, default="path/to/genes_of_interest.txt",
                    help="A path to 'genes of interest' file")
parser.add_argument("-s", type=str, default="path/to/output_folder/",
                    help="A path to the output folder for edge confidence scores")

args = parser.parse_args()
file_e, file_w = args.f, args.w
file_p, file_g = args.p, args.g
save_path = (args.s).rstrip('/')

patset = read_file(file_p)
if not patset:
    print("Warning! There is no sample ID in the 'samples of interest' file.")
    sys.exit()

geneset = read_file(file_g)
if not geneset:
    print("Warning! There is no gene ID in the 'genes of interest' file.")
    sys.exit()

weight = {}
try:
    with open(file_w, mode='r') as rline:
        _ = rline.readline()
        for nline in rline:
            p, w, *_ = nline.strip('\n').split('\t')
            weight.update({p: float(w)})
except FileNotFoundError:
    print(f"Error! File not found: {file_w}")
    sys.exit()
if not weight:
    print("Warning! There is no sample ID in the 'sample weight' file.")
    sys.exit()

gene, value = [], []
try:
    with open(file_e, mode='r') as rline:
        pat = rline.readline().strip('\n').split('\t')[1:]
        for nline in rline:
            g, *v = nline.strip('\n').split('\t')
            if g in geneset:
                value += v
                gene.append(g)
except FileNotFoundError:
    print(f"Error! File not found: {file_e}")
    sys.exit()

patlen, genelen = len(pat), len(gene)
if not patlen:
    print("Warning! The 'gene expression matrix' file is empty")
    sys.exit()

patloc = [l for l, p in enumerate(pat) if p in patset]
if not genelen or len(patloc) != len(patset):
    print("Warning! The expression file cannot be mapped to 'samples of interest' or 'genes of interest' file")
    sys.exit()
if len(set(pat) & weight.keys()) != patlen:
    print("Warning! The sample ID(s) in the expression file cannot be mapped to 'sample weight' file")
    sys.exit()
print(f"patient : {len(patloc)}\ngene : {genelen}")

gene = np.array(gene)
value = np.array(value).reshape(genelen, patlen)
value = check_file(value)
value = value.astype(float)

loc = np.where(np.sum(value, axis=1) == 0)
if len(loc[0]) != 0:
    tem = ','.join(str(i) for i in gene[loc])
    print('Processing: delete gene(s) with zero expression values in all samples:' + tem)
    value = np.delete(value, loc, 0)
    gene = np.delete(gene, loc)

agg = np.corrcoef(value)

for l in patloc:
    p = pat[l]
    value_s = np.c_[value, value[:, l]]
    value_s = np.corrcoef(value_s)
    value_s = weight[p] * (value_s - agg) + agg
    df = pd.DataFrame(value_s, index=gene, columns=gene)
    df.to_csv(f"{save_path}/{p}.csv")

print("Finish")
