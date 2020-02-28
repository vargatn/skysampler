import numpy as np
import skysampler.indexer as indexer
import skysampler.paths as paths
import glob

import argparse

parser = argparse.ArgumentParser(description='Runs MultiIndexer in parallel')
parser.add_argument("--ibin", type=int, default=-1)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")

_survey_fnames_expr = "/e/ocean1/users/vargatn/DES/Y3_DATA/DES_Y3_GOLD_MOF_base*h5"
survey_fnames = np.sort(glob.glob(_survey_fnames_expr))

clust_path = "/e/ocean1/users/vargatn/EMULATOR/DELTA/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit"
rands_path = "/e/ocean1/users/vargatn/EMULATOR/DELTA/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit"

tag = "multi-indexer-epsilon_narrow-z_v001"

NPROC = 150

clust_tag = tag + "clust_"
rands_tag = tag + "rands_"
work_dir = "/e/eser2/vargatn/EMULATOR/EPSILON/indexer/"

redshift_bins = [[0.3, 0.35], [0.45, 0.5], [0.6, 0.65]]
lambda_bins = [[55, 60],]


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.noclust:
        print("starting clusters")
        i = 0
        for z, zbin in enumerate(redshift_bins):
            for l, lbin in enumerate(lambda_bins):
                if args.ibin == -1 or args.ibin == i:
                    print("z", z, "lambda", l)
                    fname_root = work_dir + clust_tag + "z" + str(z) + "_l" + str(l)
                    # print(fname_root)
                    print(fname_root)
                    target = indexer.TargetData(clust_path, mode="clust")
                    pars = ["redshift", "richness"]
                    limits = [zbin, lbin]
                    target.select_range(pars, limits)

                    survey = indexer.SurveyData(survey_fnames)

                    imaker = indexer.MultiIndexer(survey, target, fname_root)
                    imaker.run(nprocess=NPROC)
                    raise KeyboardInterrupt

                i += 1

    # if not args.norands:
    #     print("starting randoms")
    #     i = 0
    #     for z, zbin in enumerate(redshift_bins):
    #         for l, lbin in enumerate(lambda_bins):
    #             if args.ibin == -1 or args.ibin == i:
    #                 print("z", z, "lambda", l)
    #                 fname_root = work_dir + rands_tag + "_z" + str(z) + "_l" + str(l)
    #                 print(fname_root)
    #                 random = indexer.TargetData(rands_path, mode="rands")
    #                 pars = ["redshift", "richness"]
    #                 limits = [zbin, lbin]
    #                 random.select_range(pars, limits)
    #                 random.draw_subset(3000)
    #
    #                 survey = indexer.SurveyData(survey_fnames)
    #                 imaker = indexer.MultiIndexer(survey, random, fname_root)
    #                 imaker.run(nprocess=NPROC)
    #             i += 1
