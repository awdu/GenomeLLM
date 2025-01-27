# GenomeLLM

#CONTENTS
1) The scripts directory contains scripts for running SegmentNT and Grover on the cluster. For 
both, there are two scripts: the main python script (GROVER.py and SegmentNT.py) and the wrapper 
shell script (RunGrover.sh and RunSegmentNT.sh).

The wrapper shell script is used to create a slurm job which runs the python script. The
wrapper script loads a container with the necessary python packages, so it is portable to 
any cluster using slurm.

The python scripts use pytorch to run the model, so a GPU is necessary to run this code.

Documentation for GROVER can be found here: https://huggingface.co/PoetschLab/GROVER
Documentation for SegmentNT can be found here: https://huggingface.co/InstaDeepAI/segment_nt

2) The datasets directory contains the 18 datasets used in the SegmentNT paper to benchamrk 
performance. This includes:
	enhancers
	enhancers_types
	H2AFZ
	H3K27ac
	H3K27me3
	H3K36me3
	H3K4me1
	H3K4me2
	H3K4me3
	H3K9ac
	H3K9me3
	H4K20me1
	promoter_all
	promoter_no_tata
	promoter_tata
	splice_sites_acceptors
	splice_sites_all
	splice_sites_donors

There is also a script MakeDatasetTSV.py which will split the datasets into training and test 
sets.
