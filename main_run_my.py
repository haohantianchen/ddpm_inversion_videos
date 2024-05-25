import argparse
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
import os

from idna import decode
from prompt_to_prompt.ptp_classes import AttentionStore, AttentionReplace, AttentionRefine, EmptyControl,load_512
from prompt_to_prompt.ptp_utils import register_attention_control, text2image_ldm_stable, view_images
from ddm_inversion.inversion_utils import  inversion_forward_process, inversion_reverse_process, inversion_reverse_process_two_frames
from ddm_inversion.utils import image_grid,dataset_from_yaml

from torch import autocast, inference_mode
from ddm_inversion.ddim_inversion import ddim_inversion

import calendar
import time

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=int, default=1)
    parser.add_argument("--cfg_src", type=float, default=1)
    parser.add_argument("--cfg_tar", type=float, default=1)
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--dataset_yaml",  default="test_5_22.yaml")
    parser.add_argument("--eta", type=float, default=1)
    parser.add_argument("--mode",  default="our_inv", help="modes: our_inv,p2pinv,p2pddim,ddim")
    parser.add_argument("--skip",  type=int, default=0)
    parser.add_argument("--xa", type=float, default=0.6)
    parser.add_argument("--sa", type=float, default=0.2)
    
    args = parser.parse_args()
    full_data = dataset_from_yaml(args.dataset_yaml)

    # create scheduler
    # load diffusion model
    model_id = "/raid/lurenjie/AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5"
    # model_id = "stable_diff_local" # load local save of model (for internet problems)

    device = f"cuda:{args.device_num}"

    cfg_scale_src = args.cfg_src
    cfg_scale_tar = args.cfg_tar
    eta = args.eta # = 1
    skip = args.skip

    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)

    # load/reload model:
    ldm_stable = StableDiffusionPipeline.from_pretrained(model_id).to(device)



    # save_path = "/raid/lurenjie/DDPM_inversion/results_5_22/prog_noise_no_controlnet_no_cf_self-attention"
    save_path = "/raid/lurenjie/DDPM_inversion/results_debug"
    os.makedirs(save_path, exist_ok=True)

    for i in range(len(full_data)):
        current_image_data = full_data[i]
        image_path = current_image_data['init_img']
        prompt_src = current_image_data.get('source_prompt', "") # default empty string

        if args.mode=="p2pddim" or args.mode=="ddim":
            scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
            ldm_stable.scheduler = scheduler
        else:
            ldm_stable.scheduler = DDIMScheduler.from_config(model_id, subfolder = "scheduler")
            
        ldm_stable.scheduler.set_timesteps(args.num_diffusion_steps)

        # load image
        offsets=(0,0,0,0)
        x0 = load_512(image_path, *offsets, device)

        # vae encode image
        with autocast("cuda"), inference_mode():
            w0 = (ldm_stable.vae.encode(x0).latent_dist.mode() * 0.18215).float()

        # find Zs and wts - forward process
        if args.mode=="p2pddim" or args.mode=="ddim":
            wT = ddim_inversion(ldm_stable, w0, prompt_src, cfg_scale_src)
        else:
            wt, zs, wts = inversion_forward_process(ldm_stable, w0, etas=eta, prompt=prompt_src, cfg_scale=cfg_scale_src, prog_bar=True, num_inference_steps=args.num_diffusion_steps)

        for_save_path_floader = os.path.join(save_path, "{:0>3d}".format(i) + f"_cfg_src{cfg_scale_src}_cfg_scale_tar{cfg_scale_tar}")
        os.makedirs(for_save_path_floader, exist_ok=True)

        # # 新增保存noisy latent
        # file_name = os.path.basename(image_path)
        # file_name_without_extension = os.path.splitext(file_name)[0]
        # pt_path_folder = os.path.join(save_path, file_name_without_extension)
        # os.makedirs(pt_path_folder, exist_ok=True)
        # pt_path = os.path.join(pt_path_folder, "xt_girl_512_512_50.pt")
        # torch.save(wts[args.num_diffusion_steps-skip], pt_path)

        # 可视化保存结果
        with autocast("cuda"), inference_mode():
            for idx, x in enumerate(wts):
                x_dec = ldm_stable.vae.decode(1 / 0.18215 * x.unsqueeze(0)).sample
                if x_dec.dim()<4:
                    x_dec = x_dec[None,:,:,:]
                img = image_grid(x_dec)

                for_save_path = os.path.join(for_save_path_floader, f"forward_x{idx}.png")
                img.save(for_save_path)


        # iterate over decoder prompts
        prompt_tar = prompt_src

        # Check if number of words in encoder and decoder text are equal
        src_tar_len_eq = (len(prompt_src.split(" ")) == len(prompt_tar.split(" ")))

        if args.mode=="our_inv":
            # reverse process (via Zs and wT)
            # controller = AttentionStore()
            # register_attention_control(ldm_stable, controller)
            # w0, _, ws = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar], cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)
            w0, _, ws = inversion_reverse_process_two_frames(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=[prompt_tar]*2, cfg_scales=[cfg_scale_tar], prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=None)

        elif args.mode=="p2pinv":
            # inversion with attention replace
            cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
            prompts = [prompt_src, prompt_tar]
            if src_tar_len_eq:
                controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)
            else:
                # Should use Refine for target prompts with different number of tokens
                controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=args.xa, self_replace_steps=args.sa, model=ldm_stable)

            register_attention_control(ldm_stable, controller)
            w0, _ = inversion_reverse_process(ldm_stable, xT=wts[args.num_diffusion_steps-skip], etas=eta, prompts=prompts, cfg_scales=cfg_scale_list, prog_bar=True, zs=zs[:(args.num_diffusion_steps-skip)], controller=controller)
            w0 = w0[1].unsqueeze(0)

        elif args.mode=="p2pddim" or args.mode=="ddim":
            # only z=0
            if skip != 0:
                continue
            prompts = [prompt_src, prompt_tar]
            if args.mode=="p2pddim":
                if src_tar_len_eq:
                    controller = AttentionReplace(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
                # Should use Refine for target prompts with different number of tokens
                else:
                    controller = AttentionRefine(prompts, args.num_diffusion_steps, cross_replace_steps=.8, self_replace_steps=0.4, model=ldm_stable)
            else:
                controller = EmptyControl()

            register_attention_control(ldm_stable, controller)
            # perform ddim inversion
            cfg_scale_list = [cfg_scale_src, cfg_scale_tar]
            w0, latent = text2image_ldm_stable(ldm_stable, prompts, controller, args.num_diffusion_steps, cfg_scale_list, None, wT)
            w0 = w0[1:2]
        else:
            raise NotImplementedError
        

        # 可视化保存结果
        ws = list(reversed(ws))
        with autocast("cuda"), inference_mode():
            for idx, x in enumerate(ws):
                # 把ref和tgt分开decode
                x_ref = x[:1]
                x_tgt = x[-1:]

                # x_dec = ldm_stable.vae.decode(1 / 0.18215 * x).sample
                # if x_dec.dim()<4:
                #     x_dec = x_dec[None,:,:,:]
                # img = image_grid(x_dec)
                # for_save_path = os.path.join(for_save_path_floader, f"reverse_x{idx}.png")
                # img.save(for_save_path)

                x_ref_dec = ldm_stable.vae.decode(1 / 0.18215 * x_ref).sample
                if x_ref_dec.dim()<4:
                    x_ref_dec = x_ref_dec[None,:,:,:]
                img_ref = image_grid(x_ref_dec)
                ref_save_path = os.path.join(for_save_path_floader, f"reverse_ref_x{idx}.png")
                img_ref.save(ref_save_path)

                x_tgt_dec = ldm_stable.vae.decode(1 / 0.18215 * x_tgt).sample
                if x_tgt_dec.dim()<4:
                    x_tgt_dec = x_tgt_dec[None,:,:,:]
                img_tgt = image_grid(x_tgt_dec)
                tgt_save_path = os.path.join(for_save_path_floader, f"reverse_tgt_x{idx}.png")
                img_tgt.save(tgt_save_path)


