EXPERIMENTS = {
    "full": {
        "description": "完整 RIGDNet，作为主模型。",
        "overrides": {},
    },
    "rgb_only": {
        "description": "只保留 RGB 主干，作为同框架下的 RGB-only 基线。",
        "overrides": {
            "model": {
                "use_depth_branch": False,
                "use_rectifier": False,
                "use_disagreement_refinement": False,
            },
            "training": {
                "gate_loss_weight": 0.0,
                "rectify_loss_weight": 0.0,
                "disagreement_loss_weight": 0.0,
                "uncertainty_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
            },
        },
    },
    "naive_fusion": {
        "description": "双分支直接拼接融合，不使用伪深度校正和冲突精修。",
        "overrides": {
            "model": {
                "fusion_mode": "concat",
                "use_rectifier": False,
                "use_disagreement_refinement": False,
            },
            "training": {
                "gate_loss_weight": 0.0,
                "rectify_loss_weight": 0.0,
                "disagreement_loss_weight": 0.0,
                "uncertainty_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
            },
        },
    },
    "simple_gate": {
        "description": "双分支简单门控，不使用证据分解、伪深度校正和冲突精修。",
        "overrides": {
            "model": {
                "fusion_mode": "simple_gate",
                "use_rectifier": False,
                "use_disagreement_refinement": False,
            },
            "training": {
                "gate_loss_weight": 0.0,
                "rectify_loss_weight": 0.0,
                "disagreement_loss_weight": 0.0,
                "uncertainty_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
            },
        },
    },
    "wo_rectification": {
        "description": "去掉任务驱动伪深度校正。",
        "overrides": {
            "model": {
                "use_rectifier": False,
            },
            "training": {
                "rectify_loss_weight": 0.0,
            },
        },
    },
    "wo_evidence": {
        "description": "用简单双路门控替代证据分解式三路融合。",
        "overrides": {
            "model": {
                "fusion_mode": "simple_gate",
            },
            "training": {
                "gate_loss_weight": 0.0,
                "uncertainty_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
            },
        },
    },
    "wo_refinement": {
        "description": "去掉 disagreement-map 驱动的局部残差精修。",
        "overrides": {
            "model": {
                "use_disagreement_refinement": False,
            },
            "training": {
                "disagreement_loss_weight": 0.0,
            },
        },
    },
    "wo_edge": {
        "description": "去掉边界辅助分支。",
        "overrides": {
            "model": {
                "use_edge_branch": False,
            },
            "training": {
                "edge_loss_weight": 0.0,
            },
        },
    },
    "wo_depth_aug": {
        "description": "关闭深度鲁棒增强。",
        "overrides": {
            "train_dataset": {
                "depth_drop_prob": 0.0,
                "depth_noise_std": 0.0,
                "depth_blur_prob": 0.0,
            },
        },
    },
    "depth_init_random": {
        "description": "Depth 分支随机初始化，不复用 RGB 预训练。",
        "overrides": {
            "model": {
                "depth_init_mode": "random",
            },
        },
    },
    "wo_gate_supervision": {
        "description": "去掉 gate/uncertainty/entropy 三类辅助监督。",
        "overrides": {
            "training": {
                "gate_loss_weight": 0.0,
                "uncertainty_loss_weight": 0.0,
                "entropy_loss_weight": 0.0,
            },
        },
    },
    "wo_rectify_loss": {
        "description": "保留结构，只去掉 rectifier 辅助损失。",
        "overrides": {
            "training": {
                "rectify_loss_weight": 0.0,
            },
        },
    },
    "wo_disagreement_loss": {
        "description": "保留结构，只去掉 disagreement 辅助损失。",
        "overrides": {
            "training": {
                "disagreement_loss_weight": 0.0,
            },
        },
    },
}


GROUPS = {
    "quick": [
        "full",
        "rgb_only",
        "naive_fusion",
        "wo_rectification",
        "wo_evidence",
        "wo_refinement",
    ],
    "core": [
        "full",
        "rgb_only",
        "naive_fusion",
        "simple_gate",
        "wo_rectification",
        "wo_evidence",
        "wo_refinement",
        "wo_edge",
        "wo_depth_aug",
        "depth_init_random",
    ],
    "all": list(EXPERIMENTS.keys()),
}
