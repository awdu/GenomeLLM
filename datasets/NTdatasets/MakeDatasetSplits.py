datasets = ["enhancers", "H3K27me3",  "H3K4me3", "promoter_all", "splice_sites_all", "enhancers_types", "H3K36me3", "H3K9ac", "promoter_no_tata", "splice_sites_donors", "H2AFZ", "H3K4me1", "H3K9me3", "promoter_tata", "H3K27ac", "H3K4me2", "H4K20me1", "splice_sites_acceptors"]

for dataset in datasets:
	for dataType in ["test", "train"]:
		outfileName = dataset + "/" + dataset + "_" + dataType + ".tsv"
		outfile = open(outfileName, "w")
		print(outfileName)

		outText = "chromosome\tstart\tend\tlabel\tsequence\n"
		outfile.write(outText)

		infileName = dataset + "/" + dataset + "_" + dataType + "_edit.fna"
		with open(infileName, "r") as infile:
			#get rid of empty first line
			infile.readline()
			for line in infile.readlines():
				chrom = line.split(":")[0][1:]
				line = line.split(":")[1]
				start = line.split("-")[0]
				line = line.split("-")[1]
				end = line.split("|")[0]
				line = line.split("|")[1]
				label = line.split()[0]
				sequence = line.split()[1].strip()

				outText = chrom + "\t" + start + "\t" + end + "\t" + label + "\t" + sequence + "\n"
				outfile.write(outText)
		outfile.close()
