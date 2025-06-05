class LivePortraitWrapperAnimal(LivePortraitWrapper):
    """
    Wrapper for Animal - inherits LivePortraitWrapper with animal-specific checkpoints
    """
    def __init__(self, inference_cfg: InferenceConfig):
        super().__init__(inference_cfg)  # base init for device, flags, etc.

        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)

        # Override checkpoints for animal model
        self.appearance_feature_extractor = load_model(
            inference_cfg.checkpoint_F_animal, model_config, self.device, 'appearance_feature_extractor')
        log(f'Load appearance_feature_extractor from {osp.realpath(inference_cfg.checkpoint_F_animal)} done.')

        self.motion_extractor = load_model(
            inference_cfg.checkpoint_M_animal, model_config, self.device, 'motion_extractor')
        log(f'Load motion_extractor from {osp.realpath(inference_cfg.checkpoint_M_animal)} done.')

        self.warping_module = load_model(
            inference_cfg.checkpoint_W_animal, model_config, self.device, 'warping_module')
        log(f'Load warping_module from {osp.realpath(inference_cfg.checkpoint_W_animal)} done.')

        self.spade_generator = load_model(
            inference_cfg.checkpoint_G_animal, model_config, self.device, 'spade_generator')
        log(f'Load spade_generator from {osp.realpath(inference_cfg.checkpoint_G_animal)} done.')

        if inference_cfg.checkpoint_S_animal and osp.exists(inference_cfg.checkpoint_S_animal):
            self.stitching_retargeting_module = load_model(
                inference_cfg.checkpoint_S_animal, model_config, self.device, 'stitching_retargeting_module')
            log(f'Load stitching_retargeting_module from {osp.realpath(inference_cfg.checkpoint_S_animal)} done.')
        else:
            self.stitching_retargeting_module = None

        if self.compile:
            torch._dynamo.config.suppress_errors = True
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        self.timer = Timer()
