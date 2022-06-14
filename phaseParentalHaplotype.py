#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#========================================================
#Program	:	phaseParentalHaplotype.py
#Contact	:	Qichao Lian [qlian@mpipz.mpg.de]
#Date		:	14.06.2022
#Version	:	1.0
#========================================================


import argparse, hues, datetime, os, sys
from collections import OrderedDict
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np

startTime = datetime.datetime.now()


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
	description="""This script is use to phase the haplotype of parental genome.
Version:	v1.0
Author:		qclian [qlian@mpipz.mpg.de]
""")

Input = parser.add_argument_group("input files")
Input.add_argument('-i',	dest="input_file",	help='input file of geno of damily (vcf)', type=str, required=True)
Input.add_argument('-r',	dest="chr_len",			help='chromosome length files', type=str, required=True)
Input.add_argument('-c', 	dest="centr_reg",		help='centromere regions file',	type=str, required=True)

flag_parser = parser.add_mutually_exclusive_group(required=False)
flag_parser.add_argument('-f', dest='paternal', help='paternal population',	action='store_true')
flag_parser.add_argument('-m', dest='maternal', help='maternal population',	action='store_false')
parser.set_defaults(flag=True)

Output = parser.add_argument_group("output files")
Output.add_argument('-o',	dest="output_prefix",		help='output file prefix name', type=str, required=True)

args = parser.parse_args()


def main():

	Chromosome_len = ReadChrLen(args.chr_len)
	Centromere = ReadCentroReg(args.centr_reg)

	# input modified family genotype
	Paternal_line = 9
	if args.paternal:
		hues.info("paternal population")
		print()

		FamilyModifiedGeno, ChrMarkerNum, ChrMarkerMax, Samples, SampleMarkerCov = ReadInputFile(args.input_file, Paternal_line)
	else:
		Paternal_line = 10
		hues.info("maternal population")
		print()

		FamilyModifiedGeno, ChrMarkerNum, ChrMarkerMax, Samples, SampleMarkerCov = ReadInputFile(args.input_file, Paternal_line)


	print()
	FamilyPhased, FamilyCorrectedPhased, FamilyPhasedGeno, FamilyModifiedGenoCount = PhaseMutations(ChrMarkerNum, Samples, FamilyModifiedGeno)

	print()
	FamilyCOs, FamilyCorrectedPhased = IdentifyCOs(FamilyCorrectedPhased, FamilyPhasedGeno, Samples, SampleMarkerCov, Chromosome_len, Centromere)

	print()
	RefinedFamilyCOs = RefineCOs(FamilyCOs, FamilyCorrectedPhased, FamilyPhasedGeno)

	output_file_refinedFamilyCOs = args.output_prefix + ".refinedCOs.txt"
	OutputFile(RefinedFamilyCOs, output_file_refinedFamilyCOs)





def RefineCOs(FamilyCOs, FamilyCorrectedPhased, FamilyPhasedGeno):
	hues.info("Refining COs")
	print()

	RefinedFamilyCOs = OrderedDict()
	RefinedFamilyCOs["sample,chr,index"] = ["sample", "chr", "index", "start", "stop", "pre_geno", "cur_geno"]

	for co_id, co_value in FamilyCOs.items():
		# hues.log(co_id)
		# print(co_value)

		if co_id.startswith("sample,chr,index"):
			continue
		else:
			pass

		sample_id = co_value[0]
		chr_id = co_value[1]
		co_start = int(co_value[3])
		co_stop = int(co_value[4])

		start_key = sample_id + "," + chr_id + "," + str(co_start)
		start_geno = FamilyCorrectedPhased[start_key]

		stop_key = sample_id + "," + chr_id + "," + str(co_stop)
		stop_geno = FamilyCorrectedPhased[stop_key]


		for i in range(co_start + 1, co_stop):
			cur_pos_key = sample_id + "," + chr_id + "," + str(i)
			if cur_pos_key not in FamilyPhasedGeno.keys():
				continue
			else:
				pass

			cur_pos_geno = FamilyPhasedGeno[cur_pos_key]
			cur_pos_reads_num = int(cur_pos_geno[6].split("_")[0])

			pa_haplo_allele = cur_pos_geno[3]
			mo_haplo_allele = cur_pos_geno[4]

			cur_pos_allele = [0, 0]
			if cur_pos_reads_num <= -5:
				if pa_haplo_allele == "A" or pa_haplo_allele == "T":
					cur_pos_allele = [0, 1]
				elif mo_haplo_allele == "A" or mo_haplo_allele == "T":
					cur_pos_allele = [1, 0]
				else:
					hues.warn("NOT matched allele-1!")

				if cur_pos_allele[0] == start_geno[5] and cur_pos_allele[1] == start_geno[6]:
					if i > co_start and i < co_stop:
						hues.log("#refine-start#\t" + str(co_start) + " to " + str(i))
						co_start = i
					else:
						pass
				elif cur_pos_allele[0] == stop_geno[5] and cur_pos_allele[1] == stop_geno[6]:
					if i < co_stop and i > co_start:
						hues.log("#refine-stop#\t" + str(co_stop) + " to " + str(i))
						co_stop = i
					else:
						pass
				else:
					hues.warn("NOT matched allele-2!")
			else:
				continue

		co_value[3] = co_start
		co_value[4] = co_stop
		RefinedFamilyCOs[co_id] = co_value

	return RefinedFamilyCOs


def IdentifyCOs(FamilyCorrectedPhased, FamilyPhasedGeno, Samples, SampleMarkerCov, Chromosome_len, Centromere):
	hues.info("Identifying COs")
	print()

	FamilyCOs = OrderedDict()
	FamilyCOs["sample,chr,index"] = ["sample", "chr", "index", "start", "stop", "pre_geno", "cur_geno"]

	for sample_idx, sample in Samples.items():
		samplePhased = OrderedDict()
		sampleMarkerNumChr = OrderedDict()
		chr_fir_marker_pos = OrderedDict()
		chr_last_marker_pos = OrderedDict()
		co_idx = 1
		# hues.log(sample)

		for marker_key, marker_value in FamilyCorrectedPhased.items():
			if (sample + ",") in marker_key:

				marker_pa_reads = int(marker_value[5])
				marker_mo_reads = int(marker_value[6])
				if marker_pa_reads == marker_mo_reads == 0:
					samplePhased[marker_key] = marker_value
				else:
					samplePhased[marker_key] = marker_value
					cur_chr = marker_value[1]
					cur_pos = int(marker_value[2])

					if cur_chr in chr_fir_marker_pos.keys():
						if cur_pos < int(chr_fir_marker_pos[cur_chr]):
							chr_fir_marker_pos[cur_chr] = cur_pos
						else:
							pass
					else:
						chr_fir_marker_pos[cur_chr] = cur_pos

					if cur_chr in chr_last_marker_pos.keys():
						if cur_pos > int(chr_last_marker_pos[cur_chr]):
							chr_last_marker_pos[cur_chr] = cur_pos
						else:
							pass
					else:
						chr_last_marker_pos[cur_chr] = cur_pos

					familyPhasedGeno = FamilyPhasedGeno[marker_key]
					if "AltAllele" in familyPhasedGeno[5] and "AltAllele" in familyPhasedGeno[6]:
						if cur_chr in sampleMarkerNumChr.keys():
							sampleMarkerNumChr[cur_chr] += 1
						else:
							sampleMarkerNumChr[cur_chr] = 1
					else:
						continue

		SampleFilter = False
		for cur_chr in Centromere.keys():

			# check first marker position
			if cur_chr not in chr_fir_marker_pos.keys():
				hues.warn("NO mutation markers!")
				SampleFilter = True
				break
			else:
				cur_chr_fir_marker_pos = int(chr_fir_marker_pos[cur_chr])
				if cur_chr_fir_marker_pos > int(Chromosome_len[cur_chr]) / 3:
					hues.warn("First mutation marker is out of first 1/3 of chromosome!")
					SampleFilter = True
					break

			# check last marker position
			if cur_chr not in chr_last_marker_pos.keys():
				hues.warn("NO mutation markers!")
				SampleFilter = True
				break
			else:
				cur_chr_last_marker_pos = int(chr_last_marker_pos[cur_chr])
				if cur_chr_last_marker_pos < int(Chromosome_len[cur_chr]) / 3 * 2:
					hues.warn("Last mutation marker is out of last 1/3 of chromosome!")
					SampleFilter = True
					break

			# check marker number
			if cur_chr not in sampleMarkerNumChr.keys():
				hues.warn("NO mutation markers!")
				SampleFilter = True
				break
			else:
				cur_markerNum = sampleMarkerNumChr[cur_chr]
				if cur_markerNum < 5:
					hues.warn("Too few mutation markers!")
					SampleFilter = True
					break

		if SampleFilter:
			hues.warn("filter out this sample: " + sample)
			continue
		else:
			pass

		for cur_chr in Centromere.keys():
			cur_chr_len = int(Chromosome_len[cur_chr])

			pre_geno = cur_geno = ""
			pre_geno_num = cur_geno_num = 0
			pre_pos1 = pre_pos2 = cur_pos = 0
			cur_marker_idx = 0

			continual_wt_marker_idx = 0
			continual_wt_marker_pos = 0
			continual_wt_phase = ""
			continual_wt_reads_num = 0
			CO_continual_wt_start = True
			CO_continual_wt_stop = True

			for marker_key, marker_value in samplePhased.items():
				if (cur_chr + ",") in marker_key:
					cur_marker_idx += 1

					marker_pos = int(marker_value[2])
					marker_pa_allele = marker_value[3]
					marker_mo_allele = marker_value[4]
					marker_pa_reads = int(marker_value[5])
					marker_mo_reads = int(marker_value[6])

					key_marker = sample + "_" + cur_chr + "_" + str(marker_pos)
					infos = SampleMarkerCov[key_marker]
					marker_ref_reads = int(infos.split(",")[0])
					marker_alt_reads = int(infos.split(",")[1])

					if marker_pa_reads == marker_mo_reads == 0:

						if continual_wt_reads_num < 10 and marker_pos < chr_fir_marker_pos[cur_chr] and CO_continual_wt_start:
							cur_wt_phase = ""
							if marker_pa_allele == "N" or marker_mo_allele == "N":
								continue
							elif marker_pa_allele == "G" or marker_pa_allele == "C":
								cur_wt_phase = "pa"
							else:
								cur_wt_phase = "mo"

							if cur_marker_idx == 1:
								continual_wt_phase = cur_wt_phase
								continual_wt_reads_num += marker_ref_reads
								continual_wt_marker_idx = cur_marker_idx
								continual_wt_marker_pos = marker_pos
							else:
								if (continual_wt_marker_idx + 1) == cur_marker_idx:
									if cur_wt_phase == continual_wt_phase:
										continual_wt_reads_num += marker_ref_reads
										continual_wt_marker_idx = cur_marker_idx
										continual_wt_marker_pos = marker_pos
									else:
										continual_wt_marker_idx = 0
										continual_wt_marker_pos = 0
										continual_wt_phase = ""
										continual_wt_reads_num = 0								
										CO_continual_wt_start = False
								else:
									continual_wt_marker_idx = 0
									continual_wt_marker_pos = 0
									continual_wt_phase = ""
									continual_wt_reads_num = 0
									CO_continual_wt_start = False
						elif continual_wt_reads_num < 10 and marker_pos > chr_last_marker_pos[cur_chr] and CO_continual_wt_stop:
							cur_wt_phase = ""
							if marker_pa_allele == "N" or marker_mo_allele == "N":
								continue
							elif marker_pa_allele == "G" or marker_pa_allele == "C":
								cur_wt_phase = "pa"
							else:
								cur_wt_phase = "mo"

							if continual_wt_marker_idx == 0:
								continual_wt_marker_idx = cur_marker_idx
								continual_wt_phase = cur_wt_phase
							else:
								if (continual_wt_marker_idx + 1) == cur_marker_idx:
									if cur_wt_phase != cur_geno:
										continual_wt_reads_num += marker_ref_reads
										continual_wt_marker_idx = cur_marker_idx
										continual_wt_marker_pos = marker_pos
									else:
										continual_wt_marker_idx = 0
										continual_wt_marker_pos = 0
										continual_wt_phase = ""
										continual_wt_reads_num = 0	
										CO_continual_wt_stop = False	
								else:
									continual_wt_marker_idx = 0
									continual_wt_marker_pos = 0
									continual_wt_phase = ""
									continual_wt_reads_num = 0
									CO_continual_wt_stop = False
						else:
							pass

						continue
					elif marker_pa_reads > marker_mo_reads:
						marker_geno = "pa"
					else:
						marker_geno = "mo"

					if chr_fir_marker_pos[cur_chr] == marker_pos:
						if continual_wt_marker_pos != 0 and continual_wt_reads_num >= 10 and continual_wt_phase != marker_geno and CO_continual_wt_start:
							co_key = sample + "," + cur_chr + "," + str(co_idx)
							co_value = [sample, cur_chr, co_idx, continual_wt_marker_pos, marker_pos, continual_wt_phase, marker_geno]
							FamilyCOs[co_key] = co_value
							co_idx += 1

							marker_key = sample + "," + cur_chr + "," + str(continual_wt_marker_pos)
							familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
							familyCorrectedPhased[5] = marker_mo_reads
							familyCorrectedPhased[6] = marker_pa_reads
							FamilyCorrectedPhased[marker_key] = familyCorrectedPhased
						else:
							CO_continual_wt_start = False

						continual_wt_marker_idx = 0
						continual_wt_marker_pos = 0
						continual_wt_reads_num = 0
					else:
						pass

					if cur_geno == "":
						cur_geno = marker_geno
						cur_geno_num = 1
						cur_pos = marker_pos
					else:
						if cur_geno == marker_geno:
							cur_geno_num += 1
							cur_pos = marker_pos						
						else:
							if pre_geno == "":
								pre_geno = cur_geno
								pre_geno_num = cur_geno_num
								pre_pos1 = cur_pos
								pre_pos2 = marker_pos

								cur_geno = marker_geno
								cur_geno_num = 1
								cur_pos = marker_pos
							else:
								if pre_geno_num > 1:
									if cur_geno_num > 1:
										co_key = sample + "," + cur_chr + "," + str(co_idx)
										co_value = [sample, cur_chr, co_idx, pre_pos1, pre_pos2, pre_geno, cur_geno]
										FamilyCOs[co_key] = co_value
										co_idx += 1

										pre_geno = cur_geno
										pre_geno_num = cur_geno_num
										pre_pos1 = cur_pos
										pre_pos2 = marker_pos

										cur_geno = marker_geno
										cur_geno_num = 1
										cur_pos = marker_pos
									else:
										cur_geno = pre_geno
										cur_geno_num = pre_geno_num + cur_geno_num
										cur_pos = marker_pos
										
										pre_geno = ""
										pre_geno_num = 0
										pre_pos1 = pre_pos2 = 0

								else:
									if cur_geno_num > 1:
										if pre_pos1 != 0 and (pre_pos1 < 1500000 or pre_pos2 > (cur_chr_len - 1500000)):

											if CO_continual_wt_start and pre_pos1 < 1500000:
												co_key = sample + "," + cur_chr + "," + str(co_idx-1)

												marker_key = sample + "," + cur_chr + "," + str(FamilyCOs[co_key][3])
												familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
												familyCorrectedPhased[5] = 0
												familyCorrectedPhased[6] = 0
												FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

												marker_key = sample + "," + cur_chr + "," + str(pre_pos1)
												familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
												tmp = familyCorrectedPhased[5]
												familyCorrectedPhased[5] = familyCorrectedPhased[6]
												familyCorrectedPhased[6] = tmp
												FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

												del FamilyCOs[co_key]
												co_idx -= 1
											else:
												co_key = sample + "," + cur_chr + "," + str(co_idx)
												co_value = [sample, cur_chr, co_idx, pre_pos1, pre_pos2, pre_geno, cur_geno]
												FamilyCOs[co_key] = co_value
												co_idx += 1
										else:
											pass

										pre_geno = cur_geno
										pre_geno_num = cur_geno_num
										pre_pos1 = cur_pos
										pre_pos2 = marker_pos

										cur_geno = marker_geno
										cur_geno_num = 1
										cur_pos = marker_pos
									else:
										cur_geno = pre_geno
										cur_geno_num = pre_geno_num + cur_geno_num
										cur_pos = marker_pos
										
										pre_geno = ""
										pre_geno_num = 0
										pre_pos1 = pre_pos2 = 0		

					if cur_marker_idx >= cur_markerNum:
						cur_marker_idx = 1
					cur_marker_idx += 1

				else:
					continue

			if pre_geno_num > 1:
				if cur_geno_num > 1:
					co_key = sample + "," + cur_chr + "," + str(co_idx)
					co_value = [sample, cur_chr, co_idx, pre_pos1, pre_pos2, pre_geno, cur_geno]
					FamilyCOs[co_key] = co_value
					co_idx += 1

					pre_geno = cur_geno
					pre_geno_num = cur_geno_num
					pre_pos1 = cur_pos
					pre_pos2 = marker_pos

					cur_geno = marker_geno
					cur_geno_num = 1
					cur_pos = marker_pos
				else:
					if pre_pos1 != 0 and (pre_pos1 < 1500000 or pre_pos2 > (cur_chr_len - 1500000)):
						co_key = sample + "," + cur_chr + "," + str(co_idx)
						co_value = [sample, cur_chr, co_idx, pre_pos1, pre_pos2, pre_geno, cur_geno]
						FamilyCOs[co_key] = co_value
						co_idx += 1

						pre_geno = cur_geno
						pre_geno_num = cur_geno_num
						pre_pos1 = cur_pos
						pre_pos2 = marker_pos

						cur_geno = marker_geno
						cur_geno_num = 1
						cur_pos = marker_pos
					else:
						cur_geno = pre_geno
						cur_geno_num = pre_geno_num + cur_geno_num
						cur_pos = marker_pos
										
						pre_geno = ""
						pre_geno_num = 0
						pre_pos1 = pre_pos2 = 0
			else:
				if cur_geno_num > 1:
					if pre_pos1 != 0 and (pre_pos1 < 1500000 or pre_pos2 > (cur_chr_len - 1500000)):

						if CO_continual_wt_start and pre_pos1 < 1500000:
							co_key = sample + "," + cur_chr + "," + str(co_idx-1)

							marker_key = sample + "," + cur_chr + "," + str(FamilyCOs[co_key][3])
							familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
							familyCorrectedPhased[5] = 0
							familyCorrectedPhased[6] = 0
							FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

							marker_key = sample + "," + cur_chr + "," + str(pre_pos1)
							familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
							tmp = familyCorrectedPhased[5]
							familyCorrectedPhased[5] = familyCorrectedPhased[6]
							familyCorrectedPhased[6] = tmp
							FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

							del FamilyCOs[co_key]
							co_idx -= 1
						else:
							co_key = sample + "," + cur_chr + "," + str(co_idx)
							co_value = [sample, cur_chr, co_idx, pre_pos1, pre_pos2, pre_geno, cur_geno]
							FamilyCOs[co_key] = co_value
							co_idx += 1

						pre_geno = cur_geno
						pre_geno_num = cur_geno_num
						pre_pos1 = cur_pos
						pre_pos2 = marker_pos

						cur_geno = marker_geno
						cur_geno_num = 1
						cur_pos = marker_pos
					else:
						cur_geno = pre_geno
						cur_geno_num = pre_geno_num + cur_geno_num
						cur_pos = marker_pos
										
						pre_geno = ""
						pre_geno_num = 0
						pre_pos1 = pre_pos2 = 0	

			if chr_last_marker_pos[cur_chr] < marker_pos and continual_wt_marker_pos != 0 and continual_wt_reads_num >= 10 and CO_continual_wt_stop:

				marker_key = sample + "," + cur_chr + "," + str(chr_last_marker_pos[cur_chr])
				marker_value = samplePhased[marker_key]
				marker_pa_reads = int(marker_value[5])
				marker_mo_reads = int(marker_value[6])
				if marker_pa_reads > marker_mo_reads:
					marker_geno = "pa"
				else:
					marker_geno = "mo"

				co_key = sample + "," + cur_chr + "," + str(co_idx-1)
				if co_key in FamilyCOs.keys() and marker_geno != FamilyCOs[co_key][6]:
					familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
					familyCorrectedPhased[5] = marker_mo_reads
					familyCorrectedPhased[6] = marker_pa_reads
					FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

					CO_continual_wt_stop = False
				else:	
					co_key = sample + "," + cur_chr + "," + str(co_idx)
					co_value = [sample, cur_chr, co_idx, chr_last_marker_pos[cur_chr], continual_wt_marker_pos, marker_geno, continual_wt_phase]
					FamilyCOs[co_key] = co_value
					co_idx += 1

					marker_key = sample + "," + cur_chr + "," + str(continual_wt_marker_pos)
					familyCorrectedPhased = FamilyCorrectedPhased[marker_key]
					familyCorrectedPhased[5] = marker_mo_reads
					familyCorrectedPhased[6] = marker_pa_reads
					FamilyCorrectedPhased[marker_key] = familyCorrectedPhased

				continual_wt_marker_idx = 0
				continual_wt_marker_pos = 0
				continual_wt_reads_num = 0
			else:
				CO_continual_wt_stop = False
		
	return FamilyCOs, FamilyCorrectedPhased


def PhaseMutations(ChrMarkerNum, Samples, FamilyModifiedGeno):
	hues.info("Phasing Mutations")
	print()

	FamilyModifiedGenoCount = OrderedDict()
	FamilyModifiedGenoCount["sample,chr,pos"] = ["sample", "chr", "pos", "mut_num", "wt_num"]

	FamilyPhased = OrderedDict()
	FamilyPhased["sample,chr,pos"] = ["sample", "chr", "pos", "pa_allele", "mo_allele", "pa_reads", "mo_reads"]
	FamilyCorrectedPhased = OrderedDict()
	FamilyCorrectedPhased["sample,chr,pos"] = ["sample", "chr", "pos", "pa_allele", "mo_allele", "pa_reads", "mo_reads"]
	FamilyPhasedGeno = OrderedDict()
	FamilyPhasedGeno["sample,chr,pos"] = ["sample", "chr", "pos", "pa_allele", "mo_allele", "geno", "geno_corrected"]

	for cur_chr, cur_num in ChrMarkerNum.items():

		pa_haplo = mo_haplo = ""
		sample_num = len(Samples)

		window_size = 10
		for i in range(1, cur_num - window_size + 2):

			# prepare for clustering
			geno_mat = np.mat(np.random.rand(sample_num, window_size))
			col_id = 0

			for j in range(i, i + window_size):
				key = cur_chr + "," + str(j)
				value = FamilyModifiedGeno[key]
				
				raw_id = 0

				for x in range(4, sample_num + 4):
					sample_reads_num = int(value[x].split("_")[0])

					if sample_reads_num > 8:
						sample_reads_num = 8
					elif sample_reads_num < -10:
						sample_reads_num = -10
					else:
						pass

					geno_mat[raw_id, col_id] = sample_reads_num
					raw_id += 1

				col_id += 1
			
			# clustering
			Z = linkage(geno_mat, 'ward')
			k = 2
			clusters = fcluster(Z, k, criterion = 'maxclust')
			
			cluster1_num = cluster2_num = 0
			for x in range(0, sample_num):
				sample_cluster = clusters[x]
				if sample_cluster == 1:
					cluster1_num += 1
				elif sample_cluster == 2:
					cluster2_num += 1
				else:
					pass
			# hues.log("cluster1:\t" + str(cluster1_num) + "\t" + "cluster2:\t" + str(cluster2_num) + "\t" + "ratio:\t" + str(cluster1_num/sample_num))

			sampleCluster = OrderedDict()
			for x in range(0, sample_num):
				cluster_id = clusters[x]
				sample = Samples[x + 4]
				sampleCluster[sample] = cluster_id

			mut_cluster = [0 for n in range(window_size)]

			# split geno info based on cluster
			vote_cluster1_alt = 0
			vote_cluster2_alt = 0
			for j in range(i, i + window_size):
				mut_num1 = wt_num1 = 0
				mut_num2 = wt_num2 = 0

				key = cur_chr + "," + str(j)
				value = FamilyModifiedGeno[key]
				ref_alelle = value[2]
				alt_alelle = value[3]

				for x in range(4, sample_num + 4):
					sample_cluster = int(sampleCluster[Samples[x]])

					sample_reads_num = int(value[x].split("_")[0])
					if sample_reads_num > 0 and sample_cluster == 1 and "AltAllele" in value[x]:
						mut_num1 += 1
					elif sample_reads_num > 0 and sample_cluster == 2 and "AltAllele" in value[x]:
						mut_num2 += 1
					elif sample_reads_num <= -3 and sample_cluster == 1:
						wt_num1 += 1
					elif sample_reads_num <= -3 and sample_cluster == 2:
						wt_num2 += 1
					else:
						pass
				# hues.log("cluster1:\tmut-" + str(mut_num1) + "\t" + "wt-" + str(wt_num1))
				# hues.log("cluster2:\tmut-" + str(mut_num2) + "\t" + "wt-" + str(wt_num2))

				# assembly haplotype
				if pa_haplo == "":
					if mut_num2 > 0 and mut_num1 > mut_num2:
						if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
							pa_haplo = pa_haplo + alt_alelle
							mo_haplo = mo_haplo + ref_alelle
							mut_cluster[j-i] = 1
						elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
							pa_haplo = pa_haplo + alt_alelle
							mo_haplo = mo_haplo + ref_alelle
							mut_cluster[j-i] = 1
						else:
							pa_haplo = pa_haplo + "N"
							mo_haplo = mo_haplo + "N"
							mut_cluster[j-i] = 0
					elif mut_num2 == 0 and mut_num1 > mut_num2:
						pa_haplo = pa_haplo + alt_alelle
						mo_haplo = mo_haplo + ref_alelle
						mut_cluster[j-i] = 1
					elif mut_num1 > 0 and mut_num2 > mut_num1:
						if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
							pa_haplo = pa_haplo + ref_alelle
							mo_haplo = mo_haplo + alt_alelle
							mut_cluster[j-i] = 2
						elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
							pa_haplo = pa_haplo + ref_alelle
							mo_haplo = mo_haplo + alt_alelle
							mut_cluster[j-i] = 2
						else:
							pa_haplo = pa_haplo + "N"
							mo_haplo = mo_haplo + "N"
							mut_cluster[j-i] = 0
					elif mut_num1 == 0 and mut_num2 > mut_num1:
						pa_haplo = pa_haplo + ref_alelle
						mo_haplo = mo_haplo + alt_alelle
						mut_cluster[j-i] = 2
					else:
						pa_haplo = pa_haplo + "N"
						mo_haplo = mo_haplo + "N"
						mut_cluster[j-i] = 0
				else:
					haplo_len = len(pa_haplo)

					if j > haplo_len and i == 1:
						if mut_num2 > 0 and mut_num1 > mut_num2:
							if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
								pa_haplo = pa_haplo + alt_alelle
								mo_haplo = mo_haplo + ref_alelle
								mut_cluster[j-i] = 1
							elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
								pa_haplo = pa_haplo + alt_alelle
								mo_haplo = mo_haplo + ref_alelle
								mut_cluster[j-i] = 1
							else:
								pa_haplo = pa_haplo + "N"
								mo_haplo = mo_haplo + "N"
								mut_cluster[j-i] = 0
						elif mut_num2 == 0 and mut_num1 > mut_num2:
							pa_haplo = pa_haplo + alt_alelle
							mo_haplo = mo_haplo + ref_alelle
							mut_cluster[j-i] = 1
						elif mut_num1 > 0 and mut_num2 > mut_num1:
							if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
								pa_haplo = pa_haplo + ref_alelle
								mo_haplo = mo_haplo + alt_alelle
								mut_cluster[j-i] = 2
							elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
								pa_haplo = pa_haplo + ref_alelle
								mo_haplo = mo_haplo + alt_alelle
								mut_cluster[j-i] = 2
							else:
								pa_haplo = pa_haplo + "N"
								mo_haplo = mo_haplo + "N"
								mut_cluster[j-i] = 0
						elif mut_num1 == 0 and mut_num2 > mut_num1:
							pa_haplo = pa_haplo + ref_alelle
							mo_haplo = mo_haplo + alt_alelle
							mut_cluster[j-i] = 2
						else:
							pa_haplo = pa_haplo + "N"
							mo_haplo = mo_haplo + "N"
							mut_cluster[j-i] = 0
					elif j > haplo_len and i != 1:
						if vote_cluster1_alt > vote_cluster2_alt:
							if mut_num2 > 0 and mut_num1 > mut_num2:
								if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
									pa_haplo = pa_haplo + alt_alelle
									mo_haplo = mo_haplo + ref_alelle
									mut_cluster[j-i] = 1
								elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
									pa_haplo = pa_haplo + alt_alelle
									mo_haplo = mo_haplo + ref_alelle
									mut_cluster[j-i] = 1
								else:
									pa_haplo = pa_haplo + "N"
									mo_haplo = mo_haplo + "N"
									mut_cluster[j-i] = 0
							elif mut_num2 == 0 and mut_num1 > mut_num2:
								pa_haplo = pa_haplo + alt_alelle
								mo_haplo = mo_haplo + ref_alelle
								mut_cluster[j-i] = 1
							elif mut_num1 > 0 and mut_num2 > mut_num1:
								if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
									pa_haplo = pa_haplo + ref_alelle
									mo_haplo = mo_haplo + alt_alelle
									mut_cluster[j-i] = 2
								elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
									pa_haplo = pa_haplo + ref_alelle
									mo_haplo = mo_haplo + alt_alelle
									mut_cluster[j-i] = 2
								else:
									pa_haplo = pa_haplo + "N"
									mo_haplo = mo_haplo + "N"
									mut_cluster[j-i] = 0
							elif mut_num1 == 0 and mut_num2 > mut_num1:
								pa_haplo = pa_haplo + ref_alelle
								mo_haplo = mo_haplo + alt_alelle
								mut_cluster[j-i] = 2
							else:
								pa_haplo = pa_haplo + "N"
								mo_haplo = mo_haplo + "N"
								mut_cluster[j-i] = 0
						else:
							# hues.warn("Change strain!")
							if mut_num2 > 0 and mut_num1 > mut_num2:
								if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
									pa_haplo = pa_haplo + ref_alelle
									mo_haplo = mo_haplo + alt_alelle
									mut_cluster[j-i] = 1
								elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
									pa_haplo = pa_haplo + ref_alelle
									mo_haplo = mo_haplo + alt_alelle
									mut_cluster[j-i] = 1
								else:
									pa_haplo = pa_haplo + "N"
									mo_haplo = mo_haplo + "N"
									mut_cluster[j-i] = 0
							elif mut_num2 == 0 and mut_num1 > mut_num2:
								pa_haplo = pa_haplo + ref_alelle
								mo_haplo = mo_haplo + alt_alelle
								mut_cluster[j-i] = 1
							elif mut_num1 > 0 and mut_num2 > mut_num1:
								if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
									pa_haplo = pa_haplo + alt_alelle
									mo_haplo = mo_haplo + ref_alelle
									mut_cluster[j-i] = 2
								elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
									pa_haplo = pa_haplo + alt_alelle
									mo_haplo = mo_haplo + ref_alelle
									mut_cluster[j-i] = 2
								else:
									pa_haplo = pa_haplo + "N"
									mo_haplo = mo_haplo + "N"
									mut_cluster[j-i] = 0
							elif mut_num1 == 0 and mut_num2 > mut_num1:
								pa_haplo = pa_haplo + alt_alelle
								mo_haplo = mo_haplo + ref_alelle
								mut_cluster[j-i] = 2
							else:
								pa_haplo = pa_haplo + "N"
								mo_haplo = mo_haplo + "N"
								mut_cluster[j-i] = 0
					else:
						pa_haplo_allele = pa_haplo[j-1]
						mo_haplo_allele = mo_haplo[j-1]

						if pa_haplo_allele != "N":
							if mut_num2 > 0 and mut_num1 > mut_num2:
								if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
									mut_cluster[j-i] = 1
									if pa_haplo_allele == "A" or pa_haplo_allele == "T":
										vote_cluster1_alt += 1
									else:
										vote_cluster2_alt += 1
								elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
									mut_cluster[j-i] = 1
									if pa_haplo_allele == "A" or pa_haplo_allele == "T":
										vote_cluster1_alt += 1
									else:
										vote_cluster2_alt += 1
								else:
									pass
							elif mut_num2 == 0 and mut_num1 > mut_num2:
								mut_cluster[j-i] = 1
								if pa_haplo_allele == "A" or pa_haplo_allele == "T":
									vote_cluster1_alt += 1
								else:
									vote_cluster2_alt += 1
							elif mut_num1 > 0 and mut_num2 > mut_num1:
								mut_cluster[j-i] = 2
								if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
									if pa_haplo_allele != "A" and pa_haplo_allele != "T":
										vote_cluster1_alt += 1
									else:
										vote_cluster2_alt += 1
								elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
									if pa_haplo_allele != "A" and pa_haplo_allele != "T":
										vote_cluster1_alt += 1
									else:
										vote_cluster2_alt += 1
								else:
									pass
							elif mut_num1 == 0 and mut_num2 > mut_num1:
								mut_cluster[j-i] = 2
								if pa_haplo_allele != "A" and pa_haplo_allele != "T":
									vote_cluster1_alt += 1
								else:
									vote_cluster2_alt += 1
							else:
								pass
							continue

						if vote_cluster1_alt > vote_cluster2_alt:
							if mut_num2 > 0 and mut_num1 > mut_num2:
								if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
									pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 1
								elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
									pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 1
								else:
									pass
							elif mut_num2 == 0 and mut_num1 > mut_num2:
								pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
								mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
								mut_cluster[j-i] = 1
							elif mut_num1 > 0 and mut_num2 > mut_num1:
								if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
									pa_haplo = pa_haplo[0:j-1] + ref_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + alt_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 2
								elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
									pa_haplo = pa_haplo[0:j-1] + ref_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + alt_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 2
								else:
									pass
							elif mut_num1 == 0 and mut_num2 > mut_num1:
								pa_haplo = pa_haplo[0:j-1] + ref_alelle + pa_haplo[j:haplo_len]
								mo_haplo = mo_haplo[0:j-1] + alt_alelle + mo_haplo[j:haplo_len]
								mut_cluster[j-i] = 2
							else:
								pass
						else:
							# hues.warn("Change strain!")
							if mut_num2 > 0 and mut_num1 > mut_num2:
								if mut_num1 / mut_num2 >= 2 and mut_num1 - mut_num2 > 1:
									pa_haplo = pa_haplo[0:j-1] + ref_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + alt_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 1
								elif mut_num1 / mut_num2 > 1 and mut_num1 / mut_num2 < 2 and mut_num1 - mut_num2 > 1 and ( ((wt_num1 > 0 and wt_num2 / wt_num1 >= 2) or (wt_num1 == 0)) and wt_num2 - wt_num1 > 1 ):
									pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 1
								else:
									pass
							elif mut_num2 == 0 and mut_num1 > mut_num2:
								pa_haplo = pa_haplo[0:j-1] + ref_alelle + pa_haplo[j:haplo_len]
								mo_haplo = mo_haplo[0:j-1] + alt_alelle + mo_haplo[j:haplo_len]
								mut_cluster[j-i] = 1
							elif mut_num1 > 0 and mut_num2 > mut_num1:
								if mut_num2 / mut_num1 >= 2 and mut_num2 - mut_num1 > 1:
									pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 2
								elif mut_num2 / mut_num1 > 1 and mut_num2 / mut_num1 < 2 and mut_num2 - mut_num1 > 1 and ( ((wt_num2 > 0 and wt_num1 / wt_num2 >= 2) or (wt_num2 == 0)) and wt_num1 - wt_num2 > 1 ):
									pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
									mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
									mut_cluster[j-i] = 2
								else:
									pass
							elif mut_num1 == 0 and mut_num2 > mut_num1:
								pa_haplo = pa_haplo[0:j-1] + alt_alelle + pa_haplo[j:haplo_len]
								mo_haplo = mo_haplo[0:j-1] + ref_alelle + mo_haplo[j:haplo_len]
								mut_cluster[j-i] = 2
							else:
								pass

				# hues.info("pa_haplo: " + pa_haplo)
				# hues.info("mo_haplo: " + mo_haplo)

			# print()
			for z in range(window_size):
				key = cur_chr + "," + str(i+z)
				value = FamilyModifiedGeno[key]

				if mut_cluster[z] == 0:

					for x in range(4, sample_num + 4):
						sample_id = Samples[x]
						sample_cluster = int(sampleCluster[sample_id])

						key_count = sample_id + "," + cur_chr + "," + str(i+z)
						value_count = [sample_id, cur_chr, i+z, value[1], value[x], 0, 0]
						FamilyModifiedGenoCount[key_count] = value_count

				elif mut_cluster[z] == 1:
					
					for x in range(4, sample_num + 4):
						sample_id = Samples[x]
						sample_cluster = int(sampleCluster[sample_id])

						key_count = sample_id + "," + cur_chr + "," + str(i+z)
						if sample_cluster == 1:
							if key_count in FamilyModifiedGenoCount.keys():
								value_count = FamilyModifiedGenoCount[key_count]
								value_count[5] += 1
							else:
								value_count = [sample_id, cur_chr, i+z, value[1], value[x], 1, 0]
						else:
							if key_count in FamilyModifiedGenoCount.keys():
								value_count = FamilyModifiedGenoCount[key_count]
								value_count[6] += 1
							else:
								value_count = [sample_id, cur_chr, i+z, value[1], value[x], 0, 1]
						FamilyModifiedGenoCount[key_count] = value_count

				else:

					for x in range(4, sample_num + 4):
						sample_id = Samples[x]
						sample_cluster = int(sampleCluster[sample_id])

						key_count = sample_id + "," + cur_chr + "," + str(i+z)
						if sample_cluster == 1:
							if key_count in FamilyModifiedGenoCount.keys():
								value_count = FamilyModifiedGenoCount[key_count]
								value_count[6] += 1
							else:
								value_count = [sample_id, cur_chr, i+z, value[1], value[x], 0, 1]
						else:
							if key_count in FamilyModifiedGenoCount.keys():
								value_count = FamilyModifiedGenoCount[key_count]
								value_count[5] += 1
							else:
								value_count = [sample_id, cur_chr, i+z, value[1], value[x], 1, 0]
						FamilyModifiedGenoCount[key_count] = value_count


		print()
		hues.info("pa_haplo_chr: " + pa_haplo)
		hues.info("mo_haplo_chr: " + mo_haplo)
		print()


		# Genotype correction
		for i in range(1, cur_num + 1):
			key = cur_chr + "," + str(i)
			value = FamilyModifiedGeno[key]
			cur_pos = value[1]
			cur_alt = value[3]

			sample_num = len(value) - 4

			# ["sample", "chr", "pos", "pa_allele", "mo_allele", "pa_reads", "mo_reads", "pa_rate"]
			for j in range(4, sample_num + 4):
				familyPhased_key = Samples[j] + "," + cur_chr + "," + cur_pos

				key_count = Samples[j] + "," + cur_chr + "," + str(i)
				value_count = FamilyModifiedGenoCount[key_count]

				pa_haplo_allele = pa_haplo[i-1]
				mo_haplo_allele = mo_haplo[i-1]

				FamilyPhasedGeno[familyPhased_key] = [Samples[j], cur_chr, cur_pos, pa_haplo_allele, mo_haplo_allele, value[j]]

				idx_count = value_count[2]
				mut_supp = value_count[5]
				wt_supp = value_count[6]
				if mut_supp == 0 and wt_supp == 0:
					sampleGeno = 0
				else:
					mut_rate = mut_supp / (mut_supp + wt_supp)
					wt_rate = wt_supp / (mut_supp + wt_supp)

					if (mut_rate >= 0.9 and idx_count <= 9 and idx_count >= 5) or (mut_rate >= 0.9 and cur_num - idx_count < 9 and cur_num - idx_count >= 4) or (mut_rate >= 0.8 and idx_count > 9 and cur_num - idx_count >= 9 and mut_supp >= 6):
						sampleGeno = 6
						if value[j].startswith("-") and ("Unknown" not in value[j]):
							if int(value[j].split("_")[0]) >= -6:
								value[j] = "6_AltAlleleCorrected"
							else:
								sampleGeno = int(value[j].split("_")[0])

						if "Unknown" in value[j]:
							value[j] = "6_AltAlleleCorrected"

					elif (wt_rate >= 0.9 and idx_count <= 9 and idx_count >= 5) or (wt_rate >= 0.9 and cur_num - idx_count < 9 and cur_num - idx_count >= 4) or (wt_rate >= 0.8 and idx_count > 9 and cur_num - idx_count >= 9 and wt_supp >= 6):
						sampleGeno = -6
						if (not value[j].startswith("-")) and ("Unknown" not in value[j]):
							if int(value[j].split("_")[0]) < 6:
								value[j] = "-6_RefAlleleCorrected"
							else:
								sampleGeno = int(value[j].split("_")[0])
						
						if "Unknown" in value[j]:
							value[j] = "-6_RefAlleleCorrected"
					else:
						if (not value[j].startswith("-")) and ("Unknown" not in value[j]) and (idx_count < 5 or cur_num - idx_count < 4):
							sampleGeno = int(value[j].split("_")[0])
						elif (not value[j].startswith("-")) and ("Unknown" not in value[j]) and (mut_rate >= 0.6 and idx_count >= 5 and cur_num - idx_count >= 4):
							sampleGeno = int(value[j].split("_")[0])
						else:
							sampleGeno = 0

				FamilyPhasedGeno[familyPhased_key].append(value[j])

				if sampleGeno > 0 and ("Unknown" not in value[j]):
					if pa_haplo_allele == "A" or pa_haplo_allele == "T":
						FamilyPhased[familyPhased_key] = [Samples[j], cur_chr, cur_pos, pa_haplo_allele, mo_haplo_allele, 1, 0]
					elif mo_haplo_allele == "A" or mo_haplo_allele == "T":
						FamilyPhased[familyPhased_key] = [Samples[j], cur_chr, cur_pos, pa_haplo_allele, mo_haplo_allele, 0, 1]
					else:
						hues.warn("NOT matched allele!")
				else:
					FamilyPhased[familyPhased_key] = [Samples[j], cur_chr, cur_pos, pa_haplo_allele, mo_haplo_allele, 0, 0]


		# window-based phase correction
		hues.info("window-based phase correction, ROUND_1")
		for i in range(4, sample_num + 4):
			# hues.log(Samples[i])
			for j in range(1, cur_num + 1):
				key = cur_chr + "," + str(j)
				value = FamilyModifiedGeno[key]
				cur_pos = value[1]

				familyPhased_key = Samples[i] + "," + cur_chr + "," + cur_pos
				if familyPhased_key in FamilyPhased.keys():
					familyPhased_value = FamilyPhased[familyPhased_key]

					if j < 3 or j > cur_num - 2:
						continue

					if int(familyPhased_value[5]) + int(familyPhased_value[6]) > 0:

						pa_haplo_wid_cnt_pre = 0
						mo_haplo_wid_cnt_pre = 0
						pa_haplo_wid_cnt_aft = 0
						mo_haplo_wid_cnt_aft = 0

						haplo_wid_cnt_pre = 0
						haplo_wid_cnt_aft = 0

						for x in range(j - 1, 0, -1):
							key_pre = cur_chr + "," + str(x)
							value_pre = FamilyModifiedGeno[key_pre]
							familyPhased_key_pre = Samples[i] + "," + cur_chr + "," + value_pre[1]
							if familyPhased_key_pre in FamilyPhased.keys():
								familyPhased_value_pre = FamilyPhased[familyPhased_key_pre]
								haplo_wid_cnt_pre += 1
								if int(familyPhased_value_pre[5]) + int(familyPhased_value_pre[6]) > 0:
									pa_haplo_wid_cnt_pre += int(familyPhased_value_pre[5])
									mo_haplo_wid_cnt_pre += int(familyPhased_value_pre[6])
								else:
									pass
							else:
								pass

							if pa_haplo_wid_cnt_pre == 2 or mo_haplo_wid_cnt_pre == 2 or (pa_haplo_wid_cnt_pre + mo_haplo_wid_cnt_pre) == 2:
								break
							else:
								continue

						for y in range(j + 1, cur_num + 1):
							key_aft = cur_chr + "," + str(y)
							value_aft = FamilyModifiedGeno[key_aft]
							familyPhased_key_aft = Samples[i] + "," + cur_chr + "," + value_aft[1]
							if familyPhased_key_aft in FamilyPhased.keys():
								familyPhased_value_aft = FamilyPhased[familyPhased_key_aft]
								haplo_wid_cnt_aft += 1
								if int(familyPhased_value_aft[5]) + int(familyPhased_value_aft[6]) > 0:
									pa_haplo_wid_cnt_aft += int(familyPhased_value_aft[5])
									mo_haplo_wid_cnt_aft += int(familyPhased_value_aft[6])
								else:
									pass
							else:
								pass

							if pa_haplo_wid_cnt_aft == 2 or mo_haplo_wid_cnt_aft == 2 or (pa_haplo_wid_cnt_aft + mo_haplo_wid_cnt_aft) == 2:
								break
							else:
								continue

						if haplo_wid_cnt_pre >= 8 and haplo_wid_cnt_aft >= 8:
							familyPhasedGeno_value = FamilyPhasedGeno[familyPhased_key]
							if "AltAlleleCorrected" in familyPhasedGeno_value[6]:

								if familyPhased_value[5] == 1 and pa_haplo_wid_cnt_pre == 2 and pa_haplo_wid_cnt_pre == pa_haplo_wid_cnt_aft:
									pass
								elif familyPhased_value[6] == 1 and mo_haplo_wid_cnt_pre == 2 and mo_haplo_wid_cnt_pre == mo_haplo_wid_cnt_aft:
									pass
								else:
									familyPhasedGeno_value[6] = familyPhasedGeno_value[5] + "_back"
									sampleGeno = int(familyPhasedGeno_value[6].split("_")[0])

									if sampleGeno > 0 and ("Unknown" not in familyPhasedGeno_value[6]):
										if familyPhased_value[3] == "A" or familyPhased_value[3] == "T":
											FamilyPhased[familyPhased_key] = [familyPhased_value[0], familyPhased_value[1], familyPhased_value[2], familyPhased_value[3], familyPhased_value[4], 1, 0]
										elif familyPhased_value[4] == "A" or familyPhased_value[4] == "T":
											FamilyPhased[familyPhased_key] = [familyPhased_value[0], familyPhased_value[1], familyPhased_value[2], familyPhased_value[3], familyPhased_value[4], 0, 1]
										else:
											hues.warn("NOT matched allele!")
									else:
										FamilyPhased[familyPhased_key] = [familyPhased_value[0], familyPhased_value[1], familyPhased_value[2], familyPhased_value[3], familyPhased_value[4], 0, 0]

							else:
								pass

						else:
							pass

					else:
						continue
				else:
					continue

		# window-based phase correction
		hues.info("window-based phase correction, ROUND_2")
		for i in range(4, sample_num + 4):
			# hues.log(Samples[i])
			for j in range(1, cur_num + 1):
				key = cur_chr + "," + str(j)
				value = FamilyModifiedGeno[key]
				cur_pos = value[1]

				familyPhased_key = Samples[i] + "," + cur_chr + "," + cur_pos
				if familyPhased_key in FamilyPhased.keys():
					familyPhased_value = FamilyPhased[familyPhased_key]
					FamilyCorrectedPhased[familyPhased_key] = familyPhased_value

					if j < 3 or j > cur_num - 2:
						continue

					if int(familyPhased_value[5]) + int(familyPhased_value[6]) > 0:

						pa_haplo_wid_cnt_pre = 0
						mo_haplo_wid_cnt_pre = 0
						pa_haplo_wid_cnt_aft = 0
						mo_haplo_wid_cnt_aft = 0

						for x in range(j - 1, 0, -1):
							key_pre = cur_chr + "," + str(x)
							value_pre = FamilyModifiedGeno[key_pre]
							familyPhased_key_pre = Samples[i] + "," + cur_chr + "," + value_pre[1]
							if familyPhased_key_pre in FamilyPhased.keys():
								familyPhased_value_pre = FamilyPhased[familyPhased_key_pre]
								if int(familyPhased_value_pre[5]) + int(familyPhased_value_pre[6]) > 0:
									pa_haplo_wid_cnt_pre += int(familyPhased_value_pre[5])
									mo_haplo_wid_cnt_pre += int(familyPhased_value_pre[6])
								else:
									pass
							else:
								pass

							if pa_haplo_wid_cnt_pre == 2 or mo_haplo_wid_cnt_pre == 2 or (pa_haplo_wid_cnt_pre + mo_haplo_wid_cnt_pre) == 2:
								break
							else:
								continue

						for y in range(j + 1, cur_num + 1):
							key_aft = cur_chr + "," + str(y)
							value_aft = FamilyModifiedGeno[key_aft]
							familyPhased_key_aft = Samples[i] + "," + cur_chr + "," + value_aft[1]
							if familyPhased_key_aft in FamilyPhased.keys():
								familyPhased_value_aft = FamilyPhased[familyPhased_key_aft]
								if int(familyPhased_value_aft[5]) + int(familyPhased_value_aft[6]) > 0:
									pa_haplo_wid_cnt_aft += int(familyPhased_value_aft[5])
									mo_haplo_wid_cnt_aft += int(familyPhased_value_aft[6])
								else:
									pass
							else:
								pass

							if pa_haplo_wid_cnt_aft == 2 or mo_haplo_wid_cnt_aft == 2 or (pa_haplo_wid_cnt_aft + mo_haplo_wid_cnt_aft) == 2:
								break
							else:
								continue

						if pa_haplo_wid_cnt_pre == 2 and pa_haplo_wid_cnt_pre == pa_haplo_wid_cnt_aft:
							if familyPhased_value[6] == 1:
								familyPhased_value[5] = 1
								familyPhased_value[6] = 0
								FamilyCorrectedPhased[familyPhased_key] = familyPhased_value
								hues.log("Correct phase: " + familyPhased_key + "\t" + ','.join(str(e) for e in familyPhased_value))
								continue
							else:
								pass
						elif mo_haplo_wid_cnt_pre == 2 and mo_haplo_wid_cnt_pre == mo_haplo_wid_cnt_aft:
							if familyPhased_value[5] == 1:
								familyPhased_value[5] = 0
								familyPhased_value[6] = 1
								FamilyCorrectedPhased[familyPhased_key] = familyPhased_value
								hues.log("Correct phase: " + familyPhased_key + "\t" + ','.join(str(e) for e in familyPhased_value))
								continue
							else:
								pass
						else:
							pass

					else:
						continue
				else:
					continue

		pa_haplo = mo_haplo = ""

	return FamilyPhased, FamilyCorrectedPhased, FamilyPhasedGeno, FamilyModifiedGenoCount


def OutputFile(output, output_file):
	CheckOutput(output_file)

	OUT = open(output_file, 'w')
	for key, value in output.items():
		OUT_str = key.split(",")[0] + "\t" + key.split(",")[1]

		length = len(value)
		for i in range(2, length):
			OUT_str = OUT_str + "\t" + str(value[i])

		OUT.write(OUT_str + "\n")


def ReadInputFile(input_file, Paternal_line):
	CheckInput(input_file)
	input = OrderedDict()
	ChrMarkerNum = OrderedDict()
	ChrMarkerMax = OrderedDict()
	Samples = OrderedDict()
	SampleMarkerCov = OrderedDict()
	MutPosFilter_num = 0

	with open(input_file, 'r') as IN:
		for line in IN:
			if line.startswith("##"):
				continue

			lines = line.strip().split("\t")
			# hues.info(str(len(lines)))

			if line.startswith("#CHROM"):
				key = "CHROM" + "," + lines[1]
				value = ["CHROM", lines[1], "ref", "alt"]
				for i in range(11, len(lines)):
					Samples[i - 7] = lines[i]
					value.append(lines[i])
				# hues.info(Samples)
				input[key] = value
				continue

			if lines[Paternal_line].startswith("0/0") or lines[Paternal_line].startswith("1/1"):
				continue

			AltAllele_num = 0
			TotalAllele_num = 0
			if len(lines) < 3:
				key = lines[0]
				value = int(lines[1])
			else:
				value = []
				for i in range(0, len(lines)):
					if i in [0,1,3,4]:
						value.append(lines[i])
					elif i > 10:
						infos = lines[i].split(":")
						info_ref = int(infos[1].split(",")[0])
						info_alt = int(infos[1].split(",")[1])
						info_dep = int(infos[2])
						TotalAllele_num += 1

						if info_dep <= 3:
							if info_ref > 0 and info_alt == 0:
								info = "-" + str(info_ref) + "_" + "RefAllele"
							elif info_alt > 0:
								info = str(info_alt) + "_" + "AltAllele"
								AltAllele_num += 1
							elif info_ref == 0 and info_alt == 0:
								info = str(info_alt) + "_" + "Unknown"
							else:
								hues.warn("Weird geno!")
						else:
							if info_alt >= 2:
								info = str(info_alt) + "_" + "AltAllele"
								AltAllele_num += 1
							elif info_alt < 2 and info_ref > info_alt and info_ref >= 5:
								info = "-" + str(info_ref) + "_" + "RefAllele"
							elif info_alt < 2 and info_ref > info_alt and info_ref < 5:
								info = str(info_alt) + "_" + "Unknown"
							elif info_alt < 2 and info_ref <= info_alt:
								info = str(info_alt) + "_" + "Unknown"
							else:
								hues.warn("Weird geno!")

						# hues.info(info)
						value.append(info)
					else:
						continue

			if AltAllele_num / TotalAllele_num > 0.05:

				if lines[0] in ChrMarkerNum.keys():
					ChrMarkerNum[lines[0]] += 1
					if int(lines[1]) > int(ChrMarkerMax[lines[0]]):
						ChrMarkerMax[lines[0]] = lines[1]
				else:
					ChrMarkerNum[lines[0]] = 1
					ChrMarkerMax[lines[0]] = int(lines[1])

				if len(lines) < 3:
					key = lines[0]
				else:
					key = lines[0] + "," + str(ChrMarkerNum[lines[0]])

					for i in range(11, len(lines)):
						sample = Samples[i - 7]
						key_marker = sample + "_" + lines[0] + "_" + str(lines[1])
						infos = lines[i].split(":")
						key_value = infos[1]
						SampleMarkerCov[key_marker] = key_value

				input[key] = value

			else:
				MutPosFilter_num += 1
				continue

	hues.log(input_file + " Input file loaded!")
	hues.info(ChrMarkerNum)
	hues.info(ChrMarkerMax)
	hues.info(str(MutPosFilter_num) + " mutations were not detected in offsprings!")
	return input, ChrMarkerNum, ChrMarkerMax, Samples, SampleMarkerCov


def ReadCentroReg(input_centro_reg):
	CheckInput(input_centro_reg)
	centromere = OrderedDict()

	with open(input_centro_reg, 'r') as IN:
		for line in IN:
			lines = line.strip().split("\t")

			region = [int(lines[1]), int(lines[2])]
			if lines[0].startswith("Chr"):
				chr = lines[0]
			else:
				chr = "Chr" + lines[0]
			centromere[chr] = region

	hues.log(str(len(centromere)) + " Chromosome centromere regions loaded!")
	return centromere


def ReadChrLen(input_chr_len):
	CheckInput(input_chr_len)
	chr_len = OrderedDict()

	with open(input_chr_len, 'r') as IN:
		for line in IN:
			lines = line.strip().split("\t")

			chr_len[lines[0]] = lines[1]

	hues.log(str(len(chr_len)) + " Chromosome length loaded!")
	return chr_len


def CheckInput(input_file):
	if os.path.exists(input_file):
		hues.log("Found input file:\t" + input_file)

		if os.path.getsize(input_file):
			pass
		else:
			hues.error("Input file:\t" + input_file + " is empty")
			sys.exit()

	else:
		hues.error("Not found input file:\t" + input_file)
		sys.exit()


def CheckOutput(output_file):
	if os.path.exists(output_file):
		hues.warn("Found output file:\t" + output_file)
		hues.warn("Remove\t" + output_file + "\t...")
		os.remove(output_file)



if __name__ == '__main__':
	main()


print()
stopTime = datetime.datetime.now()
hues.success("This script has run " + str((stopTime - startTime).seconds) + "s!")

