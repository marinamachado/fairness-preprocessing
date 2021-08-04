from aif360.algorithms.preprocessing.reweighing import Reweighing
from aif360.algorithms.preprocessing import DisparateImpactRemover,LFR,OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc import OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions\
            import load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions\
            import get_distortion_adult, get_distortion_german, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from dataset_loader import datasets_loader




def get_priv_unpriv_att(dataset_orig):
    
    attr = dataset_orig.protected_attribute_names[0]
    idx = dataset_orig.protected_attribute_names.index(attr)
    privileged_groups =  [{attr:dataset_orig.privileged_protected_attributes[idx][0]}] 
    unprivileged_groups = [{attr:dataset_orig.unprivileged_protected_attributes[idx][0]}] 

    return privileged_groups,unprivileged_groups


def main():

    dataset = datasets_loader()
    dataset.load_german_dataset()
    data = dataset.dataset
    privileged_groups,unprivileged_groups = get_priv_unpriv_att(data)

    Lfr = LFR(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    Lfr.fit(data)
    dataset_lfr = Lfr.transform(data)

    RW = Reweighing(unprivileged_groups=unprivileged_groups,
               privileged_groups=privileged_groups)
    RW.fit(data)
    dataset_RW = RW.transform(data)


    DIR = DisparateImpactRemover()
    dataset_DIR = DIR.fit_transform(data)




if __name__ == "__main__":
    main()
