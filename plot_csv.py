import numpy as np
import pdb

# data = np.genfromtxt('/export/mlrg/vthangar/Documents/SPL-DML-PyTorch/spld_dml_results/baseline/mnist_spldml_log_2.csv', delimiter='\t', skip_header=10,
#                      skip_footer=10, names=['x', 'y', 'z', 'a', 'b'])
# pdb.set_trace()

import csv
import glob
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import argparse
parser = argparse.ArgumentParser(description='Plot Results')
parser.add_argument('--title', type=str, metavar='N',
                    help='test id number to be used for filenames')
parser.add_argument('--xmax', type=float, metavar='N',
                    help='test id number to be used for filenames')
parser.add_argument('--ymin', type=float, metavar='N',
                    help='test id number to be used for filenames')
# parser.add_argument('--name', type=str, metavar='N',
#                     help='test id number to be used for filenames')
args = parser.parse_args()
# with open('plots_to_plot.txt') as f:
#     content = f.read().splitlines()

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() ) #different object reference each time
    return list_of_objects

plots = []
for line in open('plots_to_plot.txt'):
  if line.startswith('#'):
    continue
  else:
  	plots.append(line.splitlines()[0])

# spld_dml_results = glob.glob("./spld_dml_results/baseline/*.csv")
# spld_results = glob.glob("./spl_results/baseline/*.csv")
# random_results = glob.glob("./random_results/baseline/*.csv")
# all_results = spld_dml_results + spld_results + random_results

updates = init_list_of_objects(3)
train_acc = init_list_of_objects(3)
train_loss = init_list_of_objects(3)
test_acc = init_list_of_objects(3)
test_loss = init_list_of_objects(3)

# for i, filename in enumerate(plots):
# 	name = os.getcwd() + filename
# 	with open(name,'r') as csvfile:
# 		#pdb.set_trace()
# 		variables = csv.reader(csvfile, delimiter=',')
# 		row_num = 0
# 		#pdb.set_trace()
# 		for row in variables:
# 			#pdb.set_trace()
# 			if row_num > 29:

# 				updates[i].append(float(row[0]))
# 				pdb.set_trace()
# 				train_acc[i].append(float(row[1]))
# 				train_loss[i].append(float(row[2]))
# 				test_acc[i].append(float(row[3]))
# 				test_loss[i].append(float(row[4]))
# 				#pdb.set_trace()
# 	    	row_num += 1
# 		#pdb.set_trace()

# pdb.set_trace()
fnum = 0
with open(os.getcwd() + plots[fnum],'r') as csvfile:
    variables = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in variables:
    	if i >= 29:
			#pdb.set_trace()
			updates[fnum].append(float(row[0]))
			train_acc[fnum].append(float(row[1]))
			train_loss[fnum].append(float(row[2]))
			test_acc[fnum].append(float(row[3]))
			test_loss[fnum].append(float(row[4]))
    	i += 1

fnum += 1
with open(os.getcwd() + plots[fnum],'r') as csvfile:
    variables = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in variables:
    	if i >= 29:
			#pdb.set_trace()
			updates[fnum].append(float(row[0]))
			train_acc[fnum].append(float(row[1]))
			train_loss[fnum].append(float(row[2]))
			test_acc[fnum].append(float(row[3]))
			test_loss[fnum].append(float(row[4]))
    	i += 1

fnum += 1
with open(os.getcwd() + plots[fnum],'r') as csvfile:
    variables = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in variables:
    	if i >= 29:
			#pdb.set_trace()
			updates[fnum].append(float(row[0]))
			train_acc[fnum].append(float(row[1]))
			train_loss[fnum].append(float(row[2]))
			test_acc[fnum].append(float(row[3]))
			test_loss[fnum].append(float(row[4]))
    	i += 1

start= 0
min_vals = []
min_vals.append(min(updates[0]))
min_vals.append(min(updates[1]))
min_vals.append(min(updates[2]))

args.xmin = min(min_vals) - 10
args.xmin = 0.0
#pdb.set_trace()
updates[0].insert(0, start)
updates[1].insert(0, start)
updates[2].insert(0, start)
train_acc[0].insert(0, start)
train_acc[1].insert(0, start)
train_acc[2].insert(0, start)
train_loss[0].insert(0, start)
train_loss[1].insert(0, start)
train_loss[2].insert(0, start)
test_acc[0].insert(0, start)
test_acc[1].insert(0, start)
test_acc[2].insert(0, start)
test_loss[0].insert(0, start)
test_loss[1].insert(0, start)
test_loss[2].insert(0, start)

plt.plot(updates[0], train_acc[0], '-r', label="LEAP")
plt.plot(updates[1], train_acc[1], 'm--', label="SPLD")
plt.plot(updates[2], train_acc[2], 'c--', label="Random")
# plt.plot(epochs, data2[2], '-c', label="AveragedPerceptron - Test Accuracy")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='center right')
plt.xlabel("Mini-batches")
plt.ylabel("Train Accuracy (%)")
plt.xlim(xmin=args.xmin, xmax=args.xmax)
plt.ylim(ymin=args.ymin)
plt.title(args.title)
plt.savefig(args.title + "_trainacc.pdf", bbox_inches='tight')
plt.close()

plt.plot(updates[0], train_loss[0], '-r', label="LEAP")
plt.plot(updates[1], train_loss[1], 'm--', label="SPLD")
plt.plot(updates[2], train_loss[2], 'c--', label="Random")
# plt.plot(epochs, data2[2], '-c', label="AveragedPerceptron - Test Accuracy")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='center right')
plt.xlabel("Mini-batches")
plt.ylabel("Train Loss")
plt.xlim(xmin=args.xmin)
plt.ylim(ymin=0.0)
plt.title(args.title)
plt.savefig(args.title + "_trainloss.pdf", bbox_inches='tight')
plt.close()

#def plot_test(epochs, data1, data2, name):
plt.plot(updates[0], test_acc[0], '-r', label="LEAP")
plt.plot(updates[1], test_acc[1], 'm--', label="SPLD")
plt.plot(updates[2], test_acc[2], 'c--', label="Random")
# plt.plot(epochs, data2[2], '-c', label="AveragedPerceptron - Test Accuracy")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='center right')
plt.xlabel("Mini-batches")
plt.ylabel("Test Accuracy (%)")
plt.xlim(xmin=args.xmin, xmax=args.xmax)
plt.ylim(ymin=args.ymin)
plt.title(args.title)
plt.savefig(args.title + "_testacc.pdf", bbox_inches='tight')
plt.close()

plt.plot(updates[0], test_loss[0], '-r', label="LEAP")
plt.plot(updates[1], test_loss[1], 'm--', label="SPLD")
plt.plot(updates[2], test_loss[2], 'c--', label="Random")
# plt.plot(epochs, data2[2], '-c', label="AveragedPerceptron - Test Accuracy")
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.legend(loc='center right')
plt.xlabel("Mini-batches")
plt.ylabel("Test Loss")
plt.xlim(xmin=args.xmin)
plt.ylim(ymin=0.0)
plt.title(args.title)
plt.savefig(args.title + "_testloss.pdf", bbox_inches='tight')
plt.close()

#pdb.set_trace()