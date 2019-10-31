import numpy as np
import skysampler.indexer as indexer
import skysampler.paths as paths
import glob

import argparse

parser = argparse.ArgumentParser(description='Runs MultiIndexer in parallel')
parser.add_argument("--ibin", type=int, default=-1)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")

survey_fnames = np.sort(glob.glob(paths.config["catalogs"]["survey"]["wide_data_expr"]))
clust_path = paths.config["catalogs"]["targets"]["clust"]
rands_path = paths.config["catalogs"]["targets"]["rands"]

clust_tag = paths.config["tag"] + "_clust_"
rands_tag = paths.config["tag"] + "_rands_"
fname_root = paths.config["work_dir"]

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.noclust:
        print("starting clusters")
        i = 0
        for z, zbin in enumerate(paths.config["parameter_bins"]["redshift_bins"]):
            for l, lbin in enumerate(paths.config["parameter_bins"]["lambda_bins"]):
                if args.ibin == -1 or args.ibin == i:
                    print("z", z, "lambda", l)
                    fname_root = fname_root + clust_tag + "z" + str(z) + "_l" + str(l)
                    print(fname_root)
                    target = indexer.TargetData(clust_path, mode="clust")
                    pars = ["redshift", "richness"]
                    limits = [zbin, lbin]
                    target.select_range(pars, limits)

                    survey = indexer.SurveyData(survey_fnames)
                    print("HERE")
                    imaker = indexer.MultiIndexer(survey, target, fname_root)
                    imaker.run(nprocess=paths.config["nproc"])

                i += 1

    # if not args.norands:
    #     print("starting randoms")
    #     i = 0
    #     for z, zbin in enumerate(paths.config["parameter_bins"]["redshift_bins"]):
    #         for l, lbin in enumerate(paths.config["parameter_bins"]["lambda_bins"]):
    #             if args.ibin == -1 or args.ibin == i:
    #                 print("z", z, "lambda", l)
    #                 fname_root = fname_root + rands_tag + "_z" + str(z) + "_l" + str(l)
    #                 print(fname_root)
    #                 random = indexer.TargetData(rands_path, mode="rands")
    #                 pars = ["redshift", "richness"]
    #                 limits = [zbin, lbin]
    #                 random.select_range(pars, limits)
    #                 random.draw_subset(2000)
    #
    #                 survey = indexer.SurveyData(survey_fnames)
    #                 imaker = indexer.MultiIndexer(survey, random, fname_root)
    #                 imaker.run(nprocess=paths.config["nproc"])
    #             i += 1
