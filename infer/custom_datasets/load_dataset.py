def load_dataset(dataset_name):
    ### ====== T2I ====== ###
    if dataset_name == 'dpgbench':
        from .dataset_cls.t2i.dpgbench import DPGBench
        return DPGBench()
    elif dataset_name == 'genai':
        from .dataset_cls.t2i.genai import GenAI
        return GenAI()
    elif dataset_name == 'geneval':
        from .dataset_cls.t2i.geneval import GenEval
        return GenEval()
    elif dataset_name == 'geneval2':
        from .dataset_cls.t2i.geneval2 import GenEval2
        return GenEval2()
    elif dataset_name == 'hpsv2':
        from .dataset_cls.t2i.hpsv2 import HPSv2
        return HPSv2()
    elif dataset_name == 'oneig':
        from .dataset_cls.t2i.oneig import OneIG
        return OneIG()
    elif dataset_name == 't2ireasonbench':
        from .dataset_cls.t2i.t2ireasonbench import T2IReasonBench
        return T2IReasonBench()
    elif dataset_name == 'wise':
        from .dataset_cls.t2i.wise import WISE
        return WISE()
    elif dataset_name == 'unigenbench':
        from .dataset_cls.t2i.unigenbench import UniGeBench
        return UniGeBench()
    elif dataset_name == 'tiff':
        from .dataset_cls.t2i.tiff import TIIFBench
        return TIIFBench()
    elif dataset_name == 'longtext':
        from .dataset_cls.t2i.longtext import LongTextBench
        return LongTextBench()
        

    ### ====== Edit ====== ###
    elif dataset_name == 'imgedit':
        from .dataset_cls.edit.imgedit import ImgEdit
        return ImgEdit()
    
    else:
        raise ValueError(f"Benchmark {dataset_name} is not supported.")