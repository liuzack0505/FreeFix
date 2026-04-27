import argparse
import os
import torch
import numpy as np
import imageio
import tqdm
from custom.custom_refiner import Refiner
from ours.pipelines.flux_pipeline import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from ours.schedulers.flow_match_euler_discrete_scheduler import FlowMatchEulerDiscreteScheduler
from torchvision.utils import save_image
from recon.trainer import Config, save_depth_map_visualization


def refine(cfg):
    config: Config = Config(
        data_dir=cfg.base_dir,
        result_dir=cfg.output_dir,
    )

    refiner = Refiner(
        config,
        load_step=cfg.load_step,
        test_split=cfg.test_split,
        c_exp_index=cfg.c_exp_index,
        hessian_attr=cfg.hessian_attr,
        data_type=cfg.data_type,
    )

    # Using FLUX.1-dev-FP8 for refinement, which is much smaller than the original FLUX.1-dev.
    transformer = FluxTransformer2DModel.from_single_file(
        '/opt/FreeFix/models/FP8/flux1-dev-fp8.safetensors', torch_dtype=torch.bfloat16)
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", transformer=transformer, torch_dtype=torch.bfloat16)

    pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()
    # pipe = pipe.to("cuda")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config)

    output_dir = cfg.output_dir
    os.makedirs(f'{output_dir}/before_refine', exist_ok=True)
    os.makedirs(f'{output_dir}/after_refine', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/render', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/gen', exist_ok=True)
    os.makedirs(f'{output_dir}/refine/depth', exist_ok=True)
    for c_exp in cfg.c_exp_index:
        os.makedirs(f'{output_dir}/refine/masks/{c_exp}', exist_ok=True)
    before_refine_writer = imageio.get_writer(
        f'{output_dir}/before_refine.mp4', fps=12)
    gen_writer = imageio.get_writer(f'{output_dir}/refine/gen.mp4', fps=12)
    after_refine_writer = imageio.get_writer(
        f'{output_dir}/after_refine.mp4', fps=12)

    generator = torch.manual_seed(64)
    infer_steps = int(cfg.num_inference_steps * cfg.strength)
    mask_scheduler = [int(infer_steps * cfg.c_scheduler[i])
                      for i in range(len(cfg.c_scheduler))]

    # render test images before refine
    for i in range(len(refiner.test_dataset)):
        rgb, _, _, _, _, _ = refiner.render(i)
        save_image(rgb.permute(2, 0, 1),
                   f'{output_dir}/before_refine/{i:03d}.jpg')
        before_refine_writer.append_data(
            (rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    before_refine_writer.close()

    # refine
    train_cams = [refiner.train_dataset[j]
                  for j in range(len(refiner.train_dataset))]
    train_prob = [1] * len(refiner.train_dataset)
    for i in tqdm.tqdm(range(len(refiner.test_dataset)), desc='Refining views'):
        rgb, masks, alpha, depth, cam_param, _ = refiner.render(i)
        masks = torch.stack(masks)
        rgb_to_refine = rgb.permute(2, 0, 1).to(pipe.device)  # (3, H, W)
        masks = masks.to(pipe.device)
        H, W = rgb_to_refine.shape[1], rgb_to_refine.shape[2]

        save_image(rgb_to_refine, f'{output_dir}/refine/render/{i:03d}.jpg')
        save_depth_map_visualization(
            depth[..., 0].cpu().numpy(), f'{output_dir}/refine/depth/{i:03d}.jpg')
        for j in range(masks.shape[0]):
            save_image(masks[j:j+1][None, ...],
                       f'{output_dir}/refine/masks/{cfg.c_exp_index[j]}/{i:03d}.jpg')

        if i == 0:
            warp_until = -1
            warp_mask = None
            refine_steps = cfg.refine_steps * 2
        else:
            warp_until = infer_steps * cfg.warp_ratio
            warp_mask = alpha
            refine_steps = cfg.refine_steps

        refined_image = pipe(
            cfg.prompt,
            negative_prompt=cfg.negative_prompt,
            image=rgb_to_refine,
            mask=masks,
            mask_scheduler=mask_scheduler,
            guide_until=infer_steps * cfg.guide_ratio,
            warp_image=rgb_to_refine,
            warp_until=warp_until,
            warp_mask=warp_mask,
            height=H,
            width=W,
            guidance_scale=3.5,
            num_inference_steps=cfg.num_inference_steps,
            generator=generator,
            strength=cfg.strength,
        ).images[0]

        refined_image = refined_image.resize((W, H))
        torch_refined_image = torch.from_numpy(np.array(refined_image))
        ixt = cam_param["K"]
        c2w = cam_param["c2w"]
        refine_cams = [{
            "image": torch_refined_image,
            "camtoworld": c2w,
            "K": ixt,
            "Gen": True,
            "image_id": f"gen_{i}",
        }]

        refined_image.save(f'{output_dir}/refine/gen/image_{i:03d}.jpg')
        gen_writer.append_data(np.array(refined_image))

        refiner.refine(
            refine_cams, train_cams, train_prob,
            max_steps=refine_steps,
            gen_loss_weight=cfg.gen_loss_weight,
            use_affine=cfg.affine,
        )

        train_cams.append(refine_cams[0])
        train_prob.append(cfg.gen_prob)

    gen_writer.close()

    # render test images after refine
    for i in range(len(refiner.test_dataset)):
        rgb, _, _, _, _, _ = refiner.render(i)
        save_image(rgb.permute(2, 0, 1),
                   f'{output_dir}/after_refine/{i:03d}.jpg')
        after_refine_writer.append_data(
            (rgb.detach().cpu().numpy() * 255).astype(np.uint8))
    after_refine_writer.close()

    refiner.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # paths & experiment
    parser.add_argument('--base_dir', type=str, required=True,
                        help='base directory containing GS results')
    parser.add_argument('--output_dir', type=str,
                        required=True, help='directory to save results')
    # refiner
    parser.add_argument('--load_step', type=int, default=29999)
    parser.add_argument('--test_split', type=str, default='test_right')
    parser.add_argument('--data_type', type=str, default='custom')
    parser.add_argument('--c_exp_index', type=float,
                        nargs='+', default=[0.001, 0.01, 0.1])
    parser.add_argument('--hessian_attr', type=str,
                        nargs='+', default=['means'])
    # diffusion
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative_prompt', type=str,
                        default='blurry, low quality, foggy, overall gray, subtitles, incomplete, ghost image, too close to camera')
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--c_scheduler', type=float,
                        nargs='+', default=[0.3, 0.9, 1.0])
    parser.add_argument('--guide_ratio', type=float, default=1.0)
    parser.add_argument('--warp_ratio', type=float, default=0.5)
    parser.add_argument('--refine_steps', type=int, default=300)
    parser.add_argument('--gen_prob', type=float, default=0.1)
    parser.add_argument('--gen_loss_weight', type=float, default=0.2)
    parser.add_argument(
        '--affine', action=argparse.BooleanOptionalAction, default=True)
    cfg = parser.parse_args()
    refine(cfg)
