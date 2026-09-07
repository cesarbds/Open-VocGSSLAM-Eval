#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self.data_device = "cuda"
        self.percent_dense = 0.01 # 0.01
        self.eval = True
        self._white_background = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0000016 # 0.000016
        self.position_lr_final = 0.0000016 # 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05  # 0.05
        self.scaling_lr = 0.005 # 0.005
        self.rotation_lr = 0.001 # 0.001
        self.percent_dense = 0.01 # 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100 # 100
        self.opacity_reset_interval = 600 # 3000
        self.densify_from_iter = 300    #500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 0.0002
        
        # for slam
        self.per_frame_iteration = 1
        self.downsample_rate = 10
        self.viewer_fps = 10.0
        self.max_correspondence_distance = 0.05
        self.keyframe_freq = 30
        self.train = True
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    # try:
    #     cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
    #     print("Looking for config file in", cfgfilepath)
    #     with open(cfgfilepath) as cfg_file:
    #         print("Config file found: {}".format(cfgfilepath))
    #         cfgfile_string = cfg_file.read()
    # except TypeError:
    #     print("Config file not found at")
    #     pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

class SLAMParameters():
    def __init__(self):
        ## Model parameters ##
        self.sh_degree = 0  # 3
        self._dataset_path = "/home/adrolab/Documentos/Cesar/Open-VocGSSLAM/Replica"
        self._save_path = "/home/adrolab/Documentos/Cesar/Open-VocGSSLAM/saved_results"
        self._model_path = ""
        self.dataset = "replica" # "tum
        self.scene_id = "room0"
        self._images = "images"
        self.save_results = True
        self._resolution = 0    # 4
        self.white_background = False
        self.device = "cuda"
        self.eval = False
        self.start_frame = 0
        self.end_frame = 2000
        self.stride = 1
        self.depth_trunc = 10
        ## Semantic extraction parameters ##

        #Speedy-splati prunning parameters
        self.prune_from_iter = 500
        self.prune_until_iter = 15_000
        self.prune_interval = 300
        self.soft_prune_ratio = 0.8
        self.importance_interval = 50
        # "score": importance-based pruning plus normal opacity/size pruning.
        # "simple": normal opacity/size pruning only.
        self.pruning_mode = "score"

        self.include_feature = True
        self.use_semantics_in_mapping = False
        self.use_semantics_gt = False
        self.type_semantic_extractor = "langsplat"#"dr_splat" # "raw", "quantized", "dr_splat", "conceptfusion"
        self.rerun_viewer = False
        self.use_pq = False
        self.clip_model = "ViT-B-16" # "ViT-B-32", "ViT-L-14", "ViT-H-14"
        self.clip_checkpoint = "open_clip/checkpoints/laion2b_s32b_b79k.bin"

        self.seg_model= "sam"
        self.sam_checkpoint = "sam2/checkpoints/sam_vit_h_4b8939.pth"
        self.sam_type = "vit_h" # "vit_b", "vit_l", "vit_h"
        self.sam_points_per_side = 8
        self.sam_pred_iou_thresh = 0.88
        self.sam_stability_score_thresh = 0.95
        self.sam_crop_n_layers = 0
        self.sam_point_per_batch = 144
        self.sam_min_mask_region_area = 100
        
        ## Mapping parameters ##
        self.loss_weights = {}
        self.loss_weights["im"] = 0.9
        self.loss_weights["depth"] = 0.1
        self.loss_weights["semantic"] = 0.1
        self.loss_weights["big_gaussian_reg"] = 0.05
        self.loss_weights["small_gaussian_reg"] = 0.005
        self.loss_weights["rel_rgb"] = 0.1
        self.loss_weights["rel_depth"] = 0.1
        self.loss_weights["obj_3d"] = 0.1

        self.use_l1 = True
        self.use_semantic_for_mapping = False
        self.use_reg_loss = True
        self.trackable_opacity_th = 0.05
        self.keyframe_freq = 10
        self.keyframe_miou_th = 0.7
        self.percent_dense = 0.01
        self.use_train_split = False    
        self.mapping_num_iters = 60

        ## Pipeline parameters ##
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        
        ## Optimization parameters ##
        self.iterations = 30_000
        self.position_lr_init = 0.0000016 # 0.0000016
        self.position_lr_final = 0.0000016 # 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 10_000
        self.feature_lr = 0.0025    # 0.0025
        self.opacity_lr = 0.05  # 0.05
        self.scaling_lr = 0.005 # 0.005 / best : 0.01(32.66)
        self.rotation_lr = 0.001 # 0.001
        self.percent_dense = 0.01 # 0.01
        self.semantic_lr = 0.00025
        self.language_feature_lr = 0.0025
        self.feature_lr = 0.0025
        self.lambda_dssim = 0.2
        self.densification_interval = 100 # 100
        self.opacity_reset_interval = 600 # 3000

        self.densify_from_iter = 300    #500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002 # 0.0002
        self.training_stage = 0
        

        ## ICP Parameters ##
        self.icp = {}
        self.icp["sh_degree"] = 0
        self.icp["white_background"] = False
        self.icp["knn_maxd"] = 99999.0
        self.icp["overlapped_th"] = 5e-4
        self.icp["overlapped_th2"] = 5e-5

        # for slam
        self.per_frame_iteration = 1
        self.downsample_rate = 10       # tum:5, replica:10
        self.viewer_fps = 60.0
        self.max_correspondence_distance = 0.02 # tum : 0.03, replica : 0.02
    
        self.keyframe_freq = 10 # replica : 10, tum : 10
        self.kf_threshold = 0.7
        self.train = True
        self.training_stage=0

        #langsplat
        self.vq_layer_num = 1
        self.codebook_size = 64
