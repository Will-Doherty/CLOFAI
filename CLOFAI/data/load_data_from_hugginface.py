from datasets import load_dataset

def load_hf_dset():
    dataset_dict = load_dataset("willd98/CLOFAI_test")
    return dataset_dict