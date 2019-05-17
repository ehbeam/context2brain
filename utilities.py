#!/usr/bin/python3

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams

fname = "figures/style/computer-modern/cmunss.ttf"
font = font_manager.FontProperties(fname=fname, size=22)
rcParams["axes.linewidth"] = 1.5


def load_coordinates():
	atlas_labels = pd.read_csv("data/brain/labels.csv")
	activations = pd.read_csv("data/brain/coordinates.csv", index_col=0)
	activations = activations[atlas_labels["PREPROCESSED"]].astype(float)
	return activations


def doc_mean_thres(df):
	doc_mean = df.mean()
	df_bin = 1.0 * (df.values > doc_mean.values)
	df_bin = pd.DataFrame(df_bin, columns=df.columns, index=df.index)
	return df_bin


def load_lexicon(sources):
	lexicon = []
	for source in sources:
		file = "data/text/lexicon_{}.txt".format(source)
		lexicon += [token.strip() for token in open(file, "r").readlines()]
	return sorted(lexicon)


def load_dtm(version=190325, binarize=True, sources=["cogneuro"]):
	dtm = pd.read_csv("data/text/dtm_{}.csv.gz".format(version), compression="gzip", index_col=0)
	lexicon = load_lexicon(sources)
	lexicon = sorted(list(set(lexicon).intersection(dtm.columns)))
	dtm = dtm[lexicon]
	if binarize:
		dtm = doc_mean_thres(dtm)
	return dtm.astype(float)


def load_mini_batches(X, Y, mini_batch_size=64, seed=0, reshape_labels=False):
	"""
	Creates a list of random minibatches from (X, Y)
	
	Arguments:
	X -- input data, of shape (input size, number of examples)
	Y -- true "label" vector (1 / 0), of shape (1, number of examples)
	mini_batch_size -- size of the mini-batches, integer
	
	Returns:
	mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
	"""
	
	np.random.seed(seed)			
	m = X.shape[1] # Number of training examples
	mini_batches = []
		
	# Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation]
	if reshape_labels:
		shuffled_Y = shuffled_Y.reshape((1,m))

	# Partition (shuffled_X, shuffled_Y), except the end case
	num_complete_minibatches = math.floor(m / mini_batch_size) # Mumber of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handle the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, -(m % mini_batch_size):]
		mini_batch_Y = shuffled_Y[:, -(m % mini_batch_size):]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches


def plot_loss(prefix, loss, xlab="", ylab="",
			  diag=True, alpha=0.5, color="gray"):

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the loss curve
	plt.plot(range(len(loss)), loss, alpha=alpha, 
			 c=color, linewidth=2)

	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.xlabel(xlab, fontproperties=font)
	plt.ylabel(ylab, fontproperties=font)
	ax.xaxis.set_label_coords(0.5, -0.165)
	ax.yaxis.set_label_coords(-0.275, 0.5)

	plt.savefig("figures/{}_loss.png".format(prefix), 
				bbox_inches="tight", dpi=250)
	plt.show()


def report_macro(labels, predictions):
	m = labels.size()[0]
	print("{:11s}{:4.4f}".format("F1", f1_score(labels, predictions, average="macro")))
	print("{:11s}{:4.4f}".format("Precision", precision_score(labels, predictions, average="macro")))
	print("{:11s}{:4.4f}".format("Recall", recall_score(labels, predictions, average="macro")))
	print("{:11s}{:4.4f}".format("Accuracy", (predictions == labels).sum(dim=0).float().mean().item() / m))
	print("{:11s}{:4.4f}".format("ROC-AUC", roc_auc_score(labels, predictions, average="macro")))


def report_class(labels, predictions):
	print("{:11s}{:4.4f}".format("F1", f1_score(labels, predictions, average="binary")))
	print("{:11s}{:4.4f}".format("Precision", precision_score(labels, predictions, average="binary")))
	print("{:11s}{:4.4f}".format("Recall", recall_score(labels, predictions, average="binary")))
	print("{:11s}{:4.4f}".format("Accuracy", accuracy_score(labels, predictions)))
	print("{:11s}{:4.4f}".format("ROC-AUC", roc_auc_score(labels, predictions, average=None)))


def compute_roc(labels, pred_probs):
	from sklearn.metrics import roc_curve
	fpr, tpr = [], []
	for i in range(labels.shape[1]):
		fpr_i, tpr_i, _ = roc_curve(labels[:,i], 
									pred_probs[:,i], pos_label=1)
		fpr.append(fpr_i)
		tpr.append(tpr_i)
	return fpr, tpr


def compute_prc(labels, pred_probs):
	from sklearn.metrics import precision_recall_curve
	precision, recall = [], []
	for i in range(labels.shape[1]):
		p_i, r_i, _ = precision_recall_curve(labels[:,i], 
											 pred_probs[:,i], pos_label=1)
		precision.append(p_i)
		recall.append(r_i)
	return precision, recall


def plot_curves(file_name, x, y, xlab="", ylab="",
				diag=True, alpha=0.5, color="gray"):

	fig = plt.figure(figsize=[3.6, 3.2])
	ax = fig.add_axes([0,0,1,1])

	# Plot the curves
	for i in range(len(x)):
		plt.plot(x[i], y[i], alpha=alpha, 
				 c=color, linewidth=2)

	# Plot a diagonal line
	if diag:
		plt.plot([-1,2], [-1,2], linestyle="dashed", c="k", 
				 alpha=1, linewidth=2)

	plt.xlim([-0.05, 1])
	plt.ylim([-0.05, 1])
	for side in ["right", "top"]:
		ax.spines[side].set_visible(False)
	plt.xticks(fontproperties=font)
	plt.yticks(fontproperties=font)
	ax.xaxis.set_tick_params(width=1.5, length=7)
	ax.yaxis.set_tick_params(width=1.5, length=7)

	plt.xlabel(xlab, fontproperties=font)
	plt.ylabel(ylab, fontproperties=font)
	ax.xaxis.set_label_coords(0.5, -0.165)
	ax.yaxis.set_label_coords(-0.18, 0.5)

	plt.savefig("figures/{}.png".format(file_name), 
				bbox_inches="tight", dpi=250)
	plt.show()


def load_atlas():

	import numpy as np
	from nilearn import image

	cer = "data/brain/atlases/Cerebellum-MNIfnirt-maxprob-thr0-1mm.nii.gz"
	cor = "data/brain/atlases/HarvardOxford-cort-maxprob-thr25-1mm.nii.gz"
	sub = "data/brain/atlases/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"

	sub_del_dic = {1:0, 2:0, 3:0, 12:0, 13:0, 14:0}
	sub_lab_dic_L = {4:1, 5:2, 6:3, 7:4, 9:5, 10:6, 11:7, 8:8}
	sub_lab_dic_R = {15:1, 16:2, 17:3, 18:4, 19:5, 20:6, 21:7, 7:8}

	sub_mat_L = image.load_img(sub).get_data()[91:,:,:]
	sub_mat_R = image.load_img(sub).get_data()[:91,:,:]

	for old, new in sub_del_dic.items():
		sub_mat_L[sub_mat_L == old] = new
	for old, new in sub_lab_dic_L.items():
		sub_mat_L[sub_mat_L == old] = new
	sub_mat_L = sub_mat_L + 48
	sub_mat_L[sub_mat_L == 48] = 0

	for old, new in sub_del_dic.items():
		sub_mat_R[sub_mat_R == old] = new
	for old, new in sub_lab_dic_R.items():
		sub_mat_R[sub_mat_R == old] = new
	sub_mat_R = sub_mat_R + 48
	sub_mat_R[sub_mat_R == 48] = 0

	cor_mat_L = image.load_img(cor).get_data()[91:,:,:]
	cor_mat_R = image.load_img(cor).get_data()[:91,:,:]

	mat_L = np.add(sub_mat_L, cor_mat_L)
	mat_L[mat_L > 56] = 0
	mat_R = np.add(sub_mat_R, cor_mat_R)
	mat_R[mat_R > 56] = 0

	mat_R = mat_R + 57
	mat_R[mat_R > 113] = 0
	mat_R[mat_R < 58] = 0

	cer_mat_L = image.load_img(cer).get_data()[91:,:,:]
	cer_mat_R = image.load_img(cer).get_data()[:91,:,:]
	cer_mat_L[cer_mat_L > 0] = 57
	cer_mat_R[cer_mat_R > 0] = 114

	mat_L = np.add(mat_L, cer_mat_L)
	mat_L[mat_L > 57] = 0
	mat_R = np.add(mat_R, cer_mat_R)
	mat_R[mat_R > 114] = 0

	mat = np.concatenate((mat_R, mat_L), axis=0)
	atlas_image = image.new_img_like(sub, mat)
	return atlas_image

		
def mni2vox(x, y, z):
	x = (float(x) * -1.0) + 90.0
	y = float(y) + 126.0
	z = float(z) * 2
	return (x, y, z)
	