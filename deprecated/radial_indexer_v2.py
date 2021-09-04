from __future__ import print_function
import numpy as np
import skysampler.indexer as indexer
import skysampler.paths as paths
import glob
import cPickle as pickle
import os

import argparse

parser = argparse.ArgumentParser(description='Runs MultiIndexer in parallel')
parser.add_argument("--ibin", type=int, default=-1)
parser.add_argument('--noclust', action="store_true")
parser.add_argument('--norands', action="store_true")
parser.add_argument("--convert", action="store_true")
parser.add_argument("--collate", action="store_true")


_survey_fnames_expr = "/e/ocean1/users/vargatn/DES/Y3_DATA/DES_Y3_GOLD_MOF_wide*"

clust_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/data/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_lgt20_vl02_catalog.fit"
rands_path = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/data/y3_gold_2.2.1_wide_sofcol_run2_redmapper_v6.4.22+2_randcat_z0.10-0.95_lgt020_vl02.fit"

tag = "multi-indexer-epsilon_narrow-z_zoom_high-l_v004"

NPROC = 150

clust_tag = tag + "_clust_"
rands_tag = tag + "_rands_"
work_dir = "/e/ocean1/users/vargatn/EMULATOR/EPSILON/indexer/" + tag + "/"

redshift_bins = [[0.3, 0.35], [0.45, 0.5], [0.6, 0.65]]
# lambda_bins = [[30, 45], [45, 60]]
lambda_bins = [[30, 45],]


if __name__ == "__main__":
    args = parser.parse_args()

    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)

    if args.convert:
        print("starting file_conversion...")
        survey_fnames = np.sort(glob.glob(_survey_fnames_expr + "fits"))
        indexer.convert_on_disk(survey_fnames, nprocess=NPROC)

    survey_fnames = np.sort(glob.glob(_survey_fnames_expr + "h5"))
    if not args.noclust and not args.convert:
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
                    print(limits)
                    target.select_range(pars, limits)
                    print("has", target.nrow, "targets")

                    fname_target = fname_root + "_target.p"
                    pickle.dump(target, open(fname_target, "wb"))

                    survey = indexer.SurveyData(survey_fnames)

                    imaker = indexer.MultiIndexer(survey, target, fname_root)
                    imaker.run(nprocess=NPROC)

                    fname_root = work_dir + clust_tag + "z" + str(z) + "_l" + str(l)
                    nfiles = len(np.sort(glob.glob(fname_root + "_*p")))
                    fnames = np.array([fname_root + "_" + str(i) + ".p" for i in np.arange(nfiles - 1)])
                    print(fnames)

                    fname_target = fname_root + "_target.p"
                    target = pickle.load(open(fname_target, "rb"))
                    print(fname_target)
                    print(target.nrow)

                    mdl = indexer.MultiDataLoader(fnames=fnames, force_target=target)
                    mdl.collate_samples()
                    cont = mdl.to_cont()
                    resname = fname_root + ".p"
                    print(resname)
                    pickle.dump(mdl.to_cont(), open(resname, "wb"))

                i += 1

    if not args.norands and not args.convert:
        print("starting randoms")
        i = 0
        for z, zbin in enumerate(redshift_bins):
            for l, lbin in enumerate(lambda_bins):
                if args.ibin == -1 or args.ibin == i:
                    print("z", z, "lambda", l)
                    fname_root = work_dir + rands_tag + "z" + str(z) + "_l" + str(l)
                    print(fname_root)
                    random = indexer.TargetData(rands_path, mode="rands")
                    pars = ["redshift", "richness"]
                    limits = [zbin, lbin]
                    print(limits)
                    random.select_range(pars, limits)
                    print("has", random.nrow, "targets")
                    random.draw_subset(500)

                    fname_target = fname_root + "_target.p"
                    pickle.dump(random, open(fname_target, "wb"))

                    survey = indexer.SurveyData(survey_fnames)

                    imaker = indexer.MultiIndexer(survey, random, fname_root)
                    imaker.run(nprocess=NPROC)

                    fname_root = work_dir + rands_tag + "z" + str(z) + "_l" + str(l)
                    nfiles = len(np.sort(glob.glob(fname_root + "_*p")))
                    # print(nfiles)
                    fnames = np.array([fname_root + "_" + str(i) + ".p" for i in np.arange(nfiles - 1)])
                    # print(fnames)
                    print(fnames[0])
                    fname_target = fname_root + "_target.p"
                    target = pickle.load(open(fname_target, "rb"))
                    print(fname_target)
                    print(target.nrow)

                    mdl = indexer.MultiDataLoader(fnames=fnames, force_target=target)
                    # mdl = indexer.MultiDataLoader(fnames=fnames, force_target=False)
                    mdl.collate_samples()
                    cont = mdl.to_cont()
                    resname = fname_root + ".p"
                    print(resname)
                    pickle.dump(mdl.to_cont(), open(resname, "wb"))

                i += 1
    #
    #
    # if args.collate:
    #     for z, zbin in enumerate(redshift_bins):
    #         for l, lbin in enumerate(lambda_bins):
    #             # pass
    #             fname_root = work_dir + clust_tag + "z" + str(z) + "_l" + str(l)
    #             nfiles = len(np.sort(glob.glob(fname_root + "_*p")))
    #             fnames = np.array([fname_root + "_" + str(i) + ".p" for i in np.arange(nfiles - 1)])
    #             print(fnames)
    #
    #             fname_target = fname_root + "_target.p"
    #             target = pickle.load(open(fname_target, "rb"))
    #             print(fname_target)
    #             print(target.nrow)
    #
    #             mdl = indexer.MultiDataLoader(fnames=fnames, force_target=target)
    #             mdl.collate_samples()
    #             cont = mdl.to_cont()
    #             resname = fname_root + ".p"
    #             print(resname)
    #             pickle.dump(mdl.to_cont(), open(resname, "wb"))
    #
    #
    #             fname_root = work_dir + rands_tag + "z" + str(z) + "_l" + str(l)
    #             nfiles = len(np.sort(glob.glob(fname_root + "_*p")))
    #             # print(nfiles)
    #             fnames = np.array([fname_root + "_" + str(i) + ".p" for i in np.arange(nfiles - 1)])
    #             # print(fnames)
    #             print(fnames[0])
    #             fname_target = fname_root + "_target.p"
    #             target = pickle.load(open(fname_target, "rb"))
    #             print(fname_target)
    #             print(target.nrow)
    #
    #             mdl = indexer.MultiDataLoader(fnames=fnames, force_target=target)
    #             # mdl = indexer.MultiDataLoader(fnames=fnames, force_target=False)
    #             mdl.collate_samples()
    #             cont = mdl.to_cont()
    #             resname = fname_root + ".p"
    #             print(resname)
    #             pickle.dump(mdl.to_cont(), open(resname, "wb"))
    #
