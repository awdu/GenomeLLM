import sys
from itertools import islice
import os
import time

import nucleotide_transformer

if "COLAB_TPU_ADDR" in os.environ:
    from jax.tools import colab_tpu

    colab_tpu.setup_tpu()

from Bio import SeqIO
import gzip
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt
from nucleotide_transformer.pretrained import get_pretrained_segment_nt_model

MAX_NUM_NUCLEOTIDES = int(sys.argv[1])

#configure jax
jax.config.update("jax_platform_name", "cpu")
backend = "gpu"
devices = jax.devices(backend)
#devices = jax.devices()
num_devices = len(devices)
print(f"Devices found: {devices}")

# seaborn settings
sns.set_style("whitegrid")
sns.set_context(
	"notebook",
	font_scale=1,
	rc={
		"font.size": 14,
		"axes.titlesize": 18,
		"axes.labelsize": 18,
		"xtick.labelsize": 16,
		"ytick.labelsize": 16,
		"legend.fontsize": 16,
		}
)

plt.rcParams['xtick.bottom'] = True
plt.rcParams['ytick.left'] = True

# set colors
colors = sns.color_palette("Set2").as_hex()
colors2 = sns.color_palette("husl").as_hex()

# Rearrange order of the features to match Fig.3 from the paper.
features_rearranged = [
	'protein_coding_gene',
	'lncRNA',
	'5UTR',
	'3UTR',
	'exon',
	'intron',
	'splice_donor',
	'splice_acceptor',
	'promoter_Tissue_specific',
	'promoter_Tissue_invariant',
	'enhancer_Tissue_specific',
	'enhancer_Tissue_invariant',
	'CTCF-bound',
	'polyA_signal',
]

def plot_features(
	predicted_probabilities_all,
	seq_length: int,
	features: List[str],
	order_to_plot: List[str],
	fig_width=8,
):
	"""
	Function to plot labels and predicted probabilities.

	Args:
		predicted_probabilities_all: Probabilities per genomic feature for each
			nucleotides in the DNA sequence.
		seq_length: DNA sequence length.
		feature: Genomic features to plot.
		order_to_plot: Order in which to plot the genomic features. This needs to be
			specified in order to match the order presented in the Fig.3 of the paper
		fig_width: Width of the figure
	"""

	sc = 1.8
	n_panels = 7

	# fig, axes = plt.subplots(n_panels, 1, figsize=(fig_width * sc, (n_panels + 2) * sc), height_ratios=[6] + [2] * (n_panels-1))
	_, axes = plt.subplots(n_panels, 1, figsize=(fig_width * sc, (n_panels + 4) * sc))

	for n, feat in enumerate(order_to_plot):
		feat_id = features.index(feat)
		prob_dist = predicted_probabilities_all[:, feat_id]

		# Use the appropriate subplot
		ax = axes[n // 2]

		try:
			id_color = colors[feat_id]
		except:
			id_color = colors2[feat_id - 8]
	ax.plot(
		prob_dist,
		color=id_color,
		label=feat,
		linestyle="-",
		linewidth=1.5,
	)
	ax.set_xlim(0, seq_length)
	ax.grid(False)
	ax.spines['bottom'].set_color('black')
	ax.spines['top'].set_color('black')
	ax.spines['right'].set_color('black')
	ax.spines['left'].set_color('black')

	for a in range (0,n_panels):
		axes[a].set_ylim(0, 1.05)
		axes[a].set_ylabel("Prob.")
		axes[a].legend(loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0)
		if a != (n_panels-1):
			axes[a].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

	# Set common x-axis label
	axes[-1].set_xlabel("Nucleotides")
	# axes[0].axis('off')  # Turn off the axis
	axes[n_panels-1].grid(False)
	axes[n_panels-1].tick_params(axis='y', which='both', left=True, right=False, labelleft=True, labelright=False)

	axes[0].set_title("Probabilities predicted over all genomics features", fontweight="bold")

	plt.show()

def main():
	print("starting main")
	################################################################
	#			SETUP
	################################################################
	#man_num_nucleotides must be a power of 2
	#seqLen must be max_num_nucleotides*6
	max_num_nucleotides = MAX_NUM_NUCLEOTIDES
	seqLen = max_num_nucleotides * 6

	print("Writing header")	
	outfileName = "/run/SegmentNT_output_Jax_" + str(MAX_NUM_NUCLEOTIDES) + ".csv"
	outfile = open(outfileName, "w")
	header = "SegName,Segment"
	#add 1 probability for each nucleotide
	for i in range(seqLen):
		header = header + "," + "ExonProb" + str(i)
	header = header + "\n"
	outfile.write(header)

	#man_num_nucleotides must be a power of 2
	#seqLen must be max_num_nucleotides*6
	max_num_nucleotides = MAX_NUM_NUCLEOTIDES
	seqLen = max_num_nucleotides * 6

	assert max_num_nucleotides % 4 == 0, (
	"The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by"
	"2 to the power of the number of downsampling block, i.e 4.")

	# If max_num_nucleotides is larger than what was used to train Segment-NT, the rescaling
	# factor needs to be adapted.
	if max_num_nucleotides + 1 > 5001:
		inference_rescaling_factor = (max_num_nucleotides + 1) / 2048
	else:
		inference_rescaling_factor=None

	# The parameters have already been downloaded above
	# so we do not instantiate them. However we instantiate a new forward function
	# where the context length extension needed is in effect.
	parameters, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
		model_name="segment_nt",
		rescaling_factor=inference_rescaling_factor,
		embeddings_layers_to_save=(29,),
		attention_maps_to_save=((1, 4), (7, 10)),
		max_positions=max_num_nucleotides + 1,
	)

	#print(parameters)

	#################################################################
	#			Read in data
	#################################################################
	infileName = "/run/50kb_chunks_final.fa"

	batchSize = 6
	#need to read 2 lines for each item in the batch
	chunkSize = batchSize * 2
		
	print("opening read file")
	with open(infileName, "r") as infile:
		for n_lines in iter(lambda: tuple(islice(infile, chunkSize)), ()):
			names = []
			segments = []
			for i in range(len(n_lines)):
				curr = n_lines[i].strip()
				if (i % 2 == 0):
					names.append(curr)
				else:
					#trim length to match model length
					segments.append(curr[:seqLen])

			'''
			for i in range(len(segments)):
				print("Before len:")
				print(len(segments[i]))
				segments[i] = segments[i].replace("N", "")		
				print("After len:")
				print(len(segments[i]))
			'''	

			#remove sequences that contain N
			toRemoveSeg = []
			toRemoveNames = []
			for i in range(len(segments)):
				if ("N" in segments[i]):
					toRemoveSeg.append(segments[i])
					toRemoveNames.append(names[i])
			for i in range(len(toRemoveSeg)):
				segments.remove(toRemoveSeg[i])
				names.remove(toRemoveNames[i])
			#Make sure there is at least 1 segment without an N
			if (len(segments) > 0):
				print("Running %s, with %s segments" % (MAX_NUM_NUCLEOTIDES, len(segments)))
				start_time = time.time()
				probs = RunModel(MAX_NUM_NUCLEOTIDES, parameters, segments, seqLen)
				end_time = time.time()
				print("Took %s seconds" % (end_time - start_time))

			#print(probs)
			#print(probs.shape)
			#write to outfile
			for i in range(len(names)):
				currLine = "%s,%s" % (names[i], segments[i])
				currProbs = probs[i,:]
				for idx in range(seqLen):
					currLine = currLine + "," + str(currProbs[idx])
				outfile.write(currLine)
				outfile.write("\n")

	outfile.close()




def RunModel(MAX_NUM_NUCLEOTIDES, parameters, segments, seqLen):
	max_num_nucleotides = MAX_NUM_NUCLEOTIDES

	assert max_num_nucleotides % 4 == 0, (
	"The number of DNA tokens (excluding the CLS token prepended) needs to be dividible by"
	"2 to the power of the number of downsampling block, i.e 4.")

	# If max_num_nucleotides is larger than what was used to train Segment-NT, the rescaling
	# factor needs to be adapted.
	if max_num_nucleotides + 1 > 5001:
		inference_rescaling_factor = (max_num_nucleotides + 1) / 2048
	else:
		inference_rescaling_factor=None

	# The parameters have already been downloaded above
	# so we do not instantiate them. However we instantiate a new forward function
	# where the context length extension needed is in effect.
	_, forward_fn, tokenizer, config = get_pretrained_segment_nt_model(
		model_name="segment_nt",
		rescaling_factor=inference_rescaling_factor,
		embeddings_layers_to_save=(29,),
		attention_maps_to_save=((1, 4), (7, 10)),
		max_positions=max_num_nucleotides + 1,
	)
	
	forward_fn = hk.transform(forward_fn)
	apply_fn = jax.pmap(forward_fn.apply, devices=devices, donate_argnums=(0,))

	# Put required quantities for the inference on the devices. This step is not
	# reproduced in the second inference since the quantities will already be loaded
	# on the devices !
	random_key = jax.random.PRNGKey(seed=0)
	keys = jax.device_put_replicated(random_key, devices=devices)
	parameters = jax.device_put_replicated(parameters, devices=devices)

	'''
	for i in range(len(segments)):
		print(len(segments[i]))	
		print(segments[i])
	'''	

	tokens_ids = [b[1] for b in tokenizer.batch_tokenize(segments)]
	tokens_str = [b[0] for b in tokenizer.batch_tokenize(segments)]

	# This stacks the batch so that it is repeated across the devices. This is done
	# in order to allow for replication even if one has more than one device.
	# To take advantage of the multiple devices and infer different sequences on
	# each of the devices, make sure to change this line into a reshape.
	# a reshape
	tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)[None, :]
	#tokens.shape, idx_stop

	# Infer
	outs = apply_fn(parameters, keys, tokens)

	# Obtain the logits over the genomic features
	logits = outs["logits"]
	# Transform them on probabilities
	probabilities = np.asarray(jax.nn.softmax(logits, axis=-1))[...,-1]
	#print("probabilities:")
	#print(probabilities.shape)
	
			
	#get exon probabilities
	idx_exon = config.features.index("exon")
	probabilities_exon = probabilities[0, :, :, idx_exon]
	#print("probabilities exon:")
	#print(probabilities_exon.shape)
	
	return probabilities_exon

#print("This is a test")
main()
