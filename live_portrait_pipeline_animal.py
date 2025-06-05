import warnings
warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly.")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True. Gradients will be None")

import torch
torch.backends.cudnn.benchmark = True

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import numpy as np
import os
import os.path as osp
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils import (
    Cropper, get_rotation_matrix, images2video, concat_frames, get_fps,
    add_audio_to_video, has_audio_stream, video2gif,
    _transform_img, prepare_paste_back, paste_back,
    load_image_rgb, load_video, resize_to_limit, dump, load,
    mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier
)
from .rprint import rlog as log
from .live_portrait_wrapper import LivePortraitWrapperAnimal


class LivePortraitPipelineAnimal:

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.wrapper = LivePortraitWrapperAnimal(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg, image_type='animal_face', flag_use_half_precision=inference_cfg.flag_use_half_precision)

    def make_motion_template(self, imgs, output_fps=25):
        motions = []
        for img in track(imgs, description="Making driving motion templates...", total=len(imgs)):
            kp_info = self.wrapper.get_kp_info(img)
            R = get_rotation_matrix(kp_info['pitch'], kp_info['yaw'], kp_info['roll']).cpu().numpy().astype(np.float32)
            motions.append({
                'scale': kp_info['scale'].cpu().numpy().astype(np.float32),
                'R': R,
                'exp': kp_info['exp'].cpu().numpy().astype(np.float32),
                't': kp_info['t'].cpu().numpy().astype(np.float32),
            })
        return {'n_frames': len(imgs), 'output_fps': output_fps, 'motion': motions}

    def load_source(self, path, max_dim, division):
        if not is_image(path):
            raise Exception(f"Unknown source format: {path}")
        img = load_image_rgb(path)
        return resize_to_limit(img, max_dim, division)

    def load_driving(self, path, inf_cfg):
        if is_template(path):
            log(f"Load template (no cropping/audio): {path}", style="bold green")
            template = load(path)
            output_fps = template.get('output_fps', inf_cfg.output_fps)
            if inf_cfg.flag_crop_driving_video:
                log("Warning: cropping ignored with template input.")
            return template, None, output_fps

        if osp.exists(path) and is_video(path):
            fps = int(get_fps(path))
            log(f"Load driving video: {path}, FPS={fps}")
            frames = load_video(path)
            if inf_cfg.flag_crop_driving_video:
                ret = self.cropper.crop_driving_video(frames)
                frames_cropped = ret['frame_crop_lst']
                log(f"Cropped driving video: {len(frames_cropped)} frames")
                frames_to_use = frames_cropped
            else:
                frames_to_use = frames
            resized = [cv2.resize(f, (256, 256)) for f in frames_to_use]
            template = self.make_motion_template(self.wrapper.prepare_videos(resized), output_fps=fps)
            wfp_template = remove_suffix(path) + '.pkl'
            dump(wfp_template, template)
            log(f"Saved motion template: {wfp_template}")
            return template, resized, fps

        raise Exception(f"{path} not exists or unsupported driving info types!")

    def animate(self, args: ArgumentConfig):
        inf_cfg = self.wrapper.inference_cfg
        device = self.wrapper.device
        crop_cfg = self.cropper.crop_cfg

        img_src = self.load_source(args.source, inf_cfg.source_max_dim, inf_cfg.source_division)
        template, driving_frames, fps = self.load_driving(args.driving, inf_cfg)

        pasteback_imgs = [] if (inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching) else None

        if inf_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(img_src, crop_cfg)
            if crop_info is None:
                raise Exception("No animal face detected in source image!")
            img_crop = crop_info['img_crop_256x256']
        else:
            img_crop = cv2.resize(img_src, (256, 256))

        I_s = self.wrapper.prepare_source(img_crop)
        x_s_info = self.wrapper.get_kp_info(I_s)
        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.wrapper.extract_feature_3d(I_s)
        x_s = self.wrapper.transform_keypoint(x_s_info)

        if pasteback_imgs is not None:
            mask_ori = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_src.shape[1], img_src.shape[0]))

        out_frames = []
        for i in track(range(template['n_frames']), description="🚀Animating...", total=template['n_frames']):
            motion = dct2device(template['motion'][i], device)

            R_d = motion.get('R', motion.get('R_d'))
            delta = motion['exp']
            t = motion['t']
            t[..., 2].fill_(0)  # zero tz
            scale = x_s_info['scale']

            x_d = scale * (x_c_s @ R_d + delta) + t

            if i == 0:
                x_d_0 = x_d
                motion_mult = calc_motion_multiplier(x_s, x_d_0)

            x_d_diff = (x_d - x_d_0) * motion_mult
            x_d = x_d_diff + x_s

            if inf_cfg.flag_stitching:
                x_d = self.wrapper.stitching(x_s, x_d)

            x_d = x_s + (x_d - x_s) * inf_cfg.driving_multiplier

            out = self.wrapper.warp_decode(f_s, x_s, x_d)
            I_out = self.wrapper.parse_output(out['out'])[0]
            out_frames.append(I_out)

            if pasteback_imgs is not None:
                pstbk = paste_back(I_out, crop_info['M_c2o'], img_src, mask_ori)
                pasteback_imgs.append(pstbk)

        mkdir(args.output_dir)
        concat_frames_list = concat_frames(driving_frames, [img_crop], out_frames)
        wfp_concat = osp.join(args.output_dir, f"{basename(args.source)}--{basename(args.driving)}_concat.mp4")
        images2video(concat_frames_list, wfp=wfp_concat, fps=fps)

        if (not is_template(args.driving)) and has_audio_stream(args.driving):
            wfp_audio_concat = osp.join(args.output_dir, f"{basename(args.source)}--{basename(args.driving)}_concat_with_audio.mp4")
            add_audio_to_video(wfp_concat, args.driving, wfp_audio_concat)
            os.replace(wfp_audio_concat, wfp_concat)
            log(f"Audio merged into {wfp_concat}")

        wfp_out = osp.join(args.output_dir, f"{basename(args.source)}--{basename(args.driving)}.mp4")
        if pasteback_imgs:
            images2video(pasteback_imgs, wfp=wfp_out, fps=fps)
        else:
            images2video(out_frames, wfp=wfp_out, fps=fps)

        if (not is_template(args.driving)) and has_audio_stream(args.driving):
            wfp_audio_out = osp.join(args.output_dir, f"{basename(args.source)}--{basename(args.driving)}_with_audio.mp4")
            add_audio_to_video(wfp_out, args.driving, wfp_audio_out)
            os.replace(wfp_audio_out, wfp_out)
            log(f"Audio merged into {wfp_out}")

        if 'wfp_template' in locals():
            log(f"Animated template: {wfp_template} - reuse with -d next time.", style="bold green")
        log(f"Animated video: {wfp_out}")
        log(f"Animated concat video: {wfp_concat}")

        wfp_gif = video2gif(wfp_out)
        log(f"Animated gif: {wfp_gif}")

        return wfp_out, wfp_concat, wfp_gif
