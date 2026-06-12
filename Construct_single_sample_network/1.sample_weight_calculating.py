import argparse
import numpy as np

#  Chen H H, Hsueh C W, Lee C H, et al. SWEET: a single-sample network inference 
#  method for deciphering individual features in disease[J]. Briefings in Bioinformatics, 2023, 24(2): bbad032.

parser = argparse.ArgumentParser(description="Manual")
parser.add_argument("-f", type=str, default="path/to/input.txt",
                    help="A path to 'gene expression matrix' file")
parser.add_argument("-k", type=float, default=0.1, help="balance parameter")
parser.add_argument("-s", type=str, default="path/to/output.txt",
                    help="A path to the output 'sample weight' file")

args = parser.parse_args()
file, k, save = args.f, args.k, args.s

gene, value = [], []
with open(file, mode='r') as rline:
    pat = rline.readline().strip('\n').split('\t')[1:]
    patlen = len(pat)
    for nline in rline:
        g, *v = nline.strip('\n').split('\t')
        value += v
        gene.append(g)

genelen = len(gene)
value = np.array(value, dtype=float).reshape(genelen, patlen).T

value = np.corrcoef(value)
value = (np.sum(value, axis=1) - 1) / (patlen - 1)
rmax, rmin = np.max(value), np.min(value)
dif = rmax - rmin + 0.01
value = (value - rmin + 0.01) / dif
value = value * k * patlen

with open(save, mode='w') as w_line:
    w_line.write('patient\tsample_weight\n')
    for p, v in zip(pat, value):
        w_line.write(f"{p}\t{v}\n")

print("Finish")
