import numpy as np
import skysampler.indexer as indexer
import glob

import argparse

parser = argparse.ArgumentParser(description='Runs MultiIndexer in parallel')
parser.add_argument("--ibin", type=int, default=-1)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")

zbins = ((0.2, 0.35), (0.35, 0.5), (0.5, 0.65))
lbins = ((30, 45), (45, 60))
survey_fnames = np.sort(glob.glob("/e/eser2/vargatn/DES/Y3_DATA/DES_Y3_GOLD_MOF_base*h5"))
clust_path = "/e/eser2/vargatn/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit"
rands_path = "/e/eser2/vargatn/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit"
fname_root = "/e/eser2/vargatn/DES/Y3_DATA/"

clust_tag = "multi-indexer-gamma_v001_clust_"
rands_tag = "multi-indexer-gamma_v001_rands_"

if __name__ == "__main__":
    args = parser.parse_args()
    if not args.noclust:
        print("starting clusters")
        i = 0
        for z, zbin in enumerate(zbins):
            for l, lbin in enumerate(lbins):
                if args.ibin == -1 or args.ibin == i:
                    print("z", z, "lambda", l)
                    fname_root = fname_root + clust_tag + "_z" + str(z) + "_l" + str(l)
                    print(fname_root)
                    target = indexer.TargetData(clust_path, mode="clust")
                    pars = ["redshift", "richness"]
                    limits = [zbin, lbin]
                    target.select_range(pars, limits)

                    survey = indexer.SurveyData(survey_fnames)
                    imaker = indexer.MultiIndexer(survey, target, fname_root)
                    imaker.run(nprocess=20)

                i += 1

    if not args.noclust:
        print("starting randoms")
        i = 0
        for z, zbin in enumerate(zbins):
            for l, lbin in enumerate(lbins):
                if args.ibin == -1 or args.ibin == i:
                    print("z", z, "lambda", l)
                    fname_root = fname_root + rands_tag + "_z" + str(z) + "_l" + str(l)
                    print(fname_root)
                    random = indexer.TargetData(rands_path, mode="rands")
                    pars = ["redshift", "richness"]
                    limits = [zbin, lbin]
                    random.select_range(pars, limits)
                    random.draw_subset(2000)

                    survey = indexer.SurveyData(survey_fnames)
                    imaker = indexer.MultiIndexer(survey, random, fname_root)
                    imaker.run(nprocess=20)
                i += 1
