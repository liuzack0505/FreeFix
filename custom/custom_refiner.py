import argparse
import os
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple
import json

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch.nn.functional as F
from custom.custom_dataset import Dataset, Parser
from recon import nerfview
import viser
import tqdm
import numpy as np
from plyfile import PlyData

from recon.trainer import Config, soft_sigmoid

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from einops import reduce
from ours.utils import neighbor_L1_loss


def _load_model_ply_path(base_dir: str) -> str:
    model_path = os.path.abspath(os.path.join(base_dir, "splat.ply"))
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"3DGS model file not found: {model_path}")
    return model_path


def _infer_sh_degree_from_ply_vertex(vertex) -> int:
    extra_f_names = [
        p.name for p in vertex.properties if p.name.startswith("f_rest_")
    ]
    if len(extra_f_names) % 3 != 0:
        raise ValueError(
            f"Invalid f_rest_* count: {len(extra_f_names)} (expected multiple of 3)."
        )

    sh_terms_minus_1 = len(extra_f_names) // 3
    sh_terms = sh_terms_minus_1 + 1
    sh_degree = int(np.sqrt(sh_terms) - 1)
    if (sh_degree + 1) ** 2 != sh_terms:
        raise ValueError(
            f"Cannot infer SH degree from f_rest count={len(extra_f_names)}."
        )
    return sh_degree


def _read_splat_data_from_ply(ply_path: str):
    plydata = PlyData.read(ply_path)
    vertex = plydata.elements[0]

    xyz = np.stack(
        (
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ),
        axis=1,
    ).astype(np.float32)
    opacities = np.asarray(
        vertex["opacity"], dtype=np.float32)[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1), dtype=np.float32)
    features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"], dtype=np.float32)
    features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"], dtype=np.float32)
    features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"], dtype=np.float32)

    sh_degree = _infer_sh_degree_from_ply_vertex(vertex)
    extra_f_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    features_extra = np.zeros(
        (xyz.shape[0], len(extra_f_names)), dtype=np.float32)
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(
            vertex[attr_name], dtype=np.float32)
    features_extra = features_extra.reshape(
        (features_extra.shape[0], 3, (sh_degree + 1) ** 2 - 1)
    )

    scale_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("scale_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    scales = np.zeros((xyz.shape[0], len(scale_names)), dtype=np.float32)
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(vertex[attr_name], dtype=np.float32)

    rot_names = sorted(
        [p.name for p in vertex.properties if p.name.startswith("rot")],
        key=lambda x: int(x.split("_")[-1]),
    )
    rots = np.zeros((xyz.shape[0], len(rot_names)), dtype=np.float32)
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(vertex[attr_name], dtype=np.float32)

    return xyz, features_dc, features_extra, opacities, scales, rots


def _write_splat_data_to_ply(
    out_ply_path: Path,
    xyz: np.ndarray,
    features_dc: np.ndarray,
    features_extra: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
):
    num_points = xyz.shape[0]
    sh_deg = int(np.sqrt(features_extra.shape[2] + 1) - 1)
    sh_terms = (sh_deg + 1) ** 2

    os.makedirs(out_ply_path.parent, exist_ok=True)

    if not (
        features_dc.shape[0] == num_points
        and features_extra.shape[0] == num_points
        and opacities.shape[0] == num_points
        and scales.shape[0] == num_points
        and rots.shape[0] == num_points
    ):
        raise ValueError(
            "All splat attributes must have the same first dimension.")

    num_rest = 3 * (sh_terms - 1)
    num_scales = scales.shape[1]
    num_rots = rots.shape[1]
    num_fields = 7 + num_rest + num_scales + num_rots

    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "property float opacity",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
    ]
    header += [f"property float f_rest_{i}" for i in range(num_rest)]
    header += [f"property float scale_{i}" for i in range(num_scales)]
    header += [f"property float rot_{i}" for i in range(num_rots)]
    header.append("end_header")

    chunk_size = 65536
    with open(out_ply_path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            count = end - start
            chunk = np.empty((count, num_fields), dtype=np.float32)
            chunk[:, 0:3] = xyz[start:end]
            chunk[:, 3] = opacities[start:end, 0]
            chunk[:, 4:7] = features_dc[start:end, :, 0]
            chunk[:, 7:7 +
                  num_rest] = features_extra[start:end].reshape(count, num_rest)
            chunk[:, 7 + num_rest:7 + num_rest +
                  num_scales] = scales[start:end]
            chunk[:, 7 + num_rest + num_scales:] = rots[start:end]
            chunk.tofile(f)


class Refiner:
    def __init__(
        self,
        cfg: Config,
        load_step=29999,
        test_split="test",
        c_exp_index=[0.001, 0.01, 0.1],
        hessian_attr=["mean"],
        data_type="custom",
        rasterize_bg_color: Tuple = (255, 255, 255),
    ):
        self.cfg = cfg
        self.device = "cuda"
        self.rasterize_bg_color = rasterize_bg_color

        if data_type == "custom":
            # cfg.data_dir points to base_dir containing colmap/ and model.ply
            self.parser = Parser(
                base_dir=cfg.data_dir,
                factor=cfg.data_factor,
                normalize=False,
            )
        else:
            raise NotImplementedError(f"Unknown data_type: {data_type}")

        self.train_dataset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            partition_file=cfg.partition
        )
        self.test_dataset = Dataset(
            self.parser,
            split="test",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            partition_file=cfg.partition
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale

        self.hessian_attr = hessian_attr
        self.c_exp_index = c_exp_index

        self.total_step = 0

        # Load the refine dataset
        # self.refine_dataset = Refine_Dataset(os.path.join(cfg.result_dir, "to_refine"))

        # Load the pre-optimized gaussian splats
        model_ply_path = _load_model_ply_path(cfg.data_dir)

        (
            means_np,
            features_dc_np,
            features_extra_np,
            opacities_np,
            scales_np,
            rots_np,
        ) = _read_splat_data_from_ply(model_ply_path)

        means = torch.from_numpy(means_np).float().to(self.device)
        scales = torch.from_numpy(scales_np).float().to(self.device)
        opacities = torch.from_numpy(
            opacities_np).float().squeeze(-1).to(self.device)
        quats = \
            torch.from_numpy(rots_np).float().to(self.device)  # (N, 4)
        colors_dc = torch.from_numpy(
            features_dc_np).float().squeeze(-1).to(self.device)  # (N, 3)
        colors_rest = torch.from_numpy(
            features_extra_np).float().to(self.device)  # (N, 3, S-1)
        colors = torch.cat([colors_dc.unsqueeze(
            2), colors_rest], dim=2)  # (N, 3, S)
        colors = colors.permute(0, 2, 1).contiguous()  # -> (N, SH, 3)

        N = means.shape[0]

        # build params list
        params = [
            ("means", torch.nn.Parameter(means)),
            ("scales", torch.nn.Parameter(scales)),
            ("quats", torch.nn.Parameter(quats)),
            ("opacities", torch.nn.Parameter(opacities)),
        ]

        # split SH coefs
        sh0 = colors[:, :1, :]  # (N, 1, 3)
        shN = colors[:, 1:, :]  # (N, SH-1, 3)
        params.append(("sh0", torch.nn.Parameter(sh0)))
        params.append(("shN", torch.nn.Parameter(shN)))

        self.splats = torch.nn.ParameterDict(
            {name: param for name, param in params})
        # self.splats = torch.nn.ParameterDict(torch.load(
        #     gs3d_model_ref.path, map_location=self.device)["splats"])

        affines = {}
        for i in range(len(self.test_dataset)):
            affines[f"gen_{i}"] = torch.nn.Parameter(
                torch.eye(4)[:3, :].to(self.device))  # 3x4
        self.affines = torch.nn.ParameterDict(affines)

        # init the optimizers
        self._init_optimizer()

        # Densification Strategy
        self.strategy = DefaultStrategy(
            verbose=True,
            # scene_scale=self.scene_scale,
            prune_opa=cfg.prune_opa,
            grow_grad2d=cfg.grow_grad2d,
            grow_scale3d=cfg.grow_scale3d,
            prune_scale3d=cfg.prune_scale3d,
            refine_start_iter=100,
            refine_stop_iter=5000,
            reset_every=1500,
            refine_every=200,
            absgrad=cfg.absgrad,
            revised_opacity=cfg.revised_opacity,
        )
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            normalize=True).to(self.device)

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            # c2ws = np.concatenate([self.parser.camtoworlds, self.refine_dataset.interp_c2ws], axis=0)
            # Ks = [self.parser.Ks_dict[camera_id].copy() for camera_id in self.parser.camera_ids] + \
            #      [self.refine_dataset.closest_K]*len(self.refine_dataset)
            # img_whs = [self.parser.imsize_dict[camera_id] for camera_id in self.parser.camera_ids] + \
            #           [self.refine_dataset.img_wh]*len(self.refine_dataset)
            c2ws = self.parser.camtoworlds
            Ks = [self.parser.Ks_dict[camera_id].copy()
                  for camera_id in self.parser.camera_ids]
            img_whs = [self.parser.imsize_dict[camera_id]
                       for camera_id in self.parser.camera_ids]
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="refining",
                c2ws=c2ws,
                Ks=Ks,
                img_whs=img_whs,
                scene_scale=self.scene_scale,
                result_dir=cfg.result_dir,
            )

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb_refine")

    def _init_optimizer(self):
        for param in self.splats.values():
            param.requires_grad = True

        self.optimizers = {
            'means': torch.optim.Adam([{"params": self.splats['means'], "lr": 1e-4 * self.scene_scale}]),
            'scales': torch.optim.Adam([{"params": self.splats['scales'], "lr": 5e-3}]),
            'quats': torch.optim.Adam([{"params": self.splats['quats'], "lr": 1e-3}]),
            'opacities': torch.optim.Adam([{"params": self.splats['opacities'], "lr": 5e-2}]),
            'sh0': torch.optim.Adam([{"params": self.splats['sh0'], "lr": 2.5e-3}]),
            'shN': torch.optim.Adam([{"params": self.splats['shN'], "lr": 2.5e-3/20}]),
        }

        self.affine_optimizers = {
            f"gen_{i}": torch.optim.Adam([{"params": self.affines[f"gen_{i}"], "lr": 1e-2}])
            for i in range(len(self.affines))
        }

    def add_splats(self, new_splats):
        for name, add_params in new_splats.items():
            param = self.splats[name]
            new_param = torch.nn.Parameter(
                torch.cat([param, add_params], dim=0), requires_grad=True)
            self.splats[name] = new_param
            optimizer = self.optimizers[name]
            for i in range(len(optimizer.param_groups)):
                param_state = optimizer.state[param]
                del optimizer.state[param]
                for key in param_state.keys():
                    if key != "step":
                        v = param_state[key]
                        param_state[key] = torch.cat(
                            [v, torch.zeros((len(add_params), *v.shape[1:]), device=v.device)])
                optimizer.param_groups[i]["params"] = [new_param]
                optimizer.state[new_param] = param_state

        for k, v in self.strategy_state.items():
            if isinstance(v, torch.Tensor):
                self.strategy_state[k] = torch.cat(
                    (v, torch.zeros((len(add_params), *v.shape[1:]), device=v.device)))

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        override_color: Tensor = None,
        affine: Tensor = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat(
                [self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_bg_color = torch.tensor(  # [C, 3]
            [self.rasterize_bg_color[0], self.rasterize_bg_color[1], self.rasterize_bg_color[2]], dtype=torch.float32, device=self.device).unsqueeze(0).expand(len(camtoworlds), -1) / 255.0
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors if override_color is None else override_color,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.absgrad,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            **kwargs,
            backgrounds=rasterize_bg_color,
        )
        if affine is not None:
            render_colors = render_colors @ affine[:3, :3] + affine[:3, 3]
        return render_colors, render_alphas, info

    def rasterize_splats_w_certainty(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int
    ):
        rgbs, alphas, _ = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB+ED",
        )
        depths = rgbs[..., 3:4].detach()[0]
        colors = torch.clamp(rgbs[..., :3], 0.0, 1.0).detach()[0]
        alphas = alphas.detach()[0, ..., 0]

        # render uncertainty
        rgbs[..., :3].backward(gradient=torch.ones_like(rgbs[..., :3]))
        H_per_gaussian = [self.splats[k].grad.detach() **
                          2 for k in self.hessian_attr]
        # H_per_gaussian = [self.splats[k].grad.detach() ** 2 for k in ["means", "quats", "scales"]]
        H_per_gaussian = torch.cat(H_per_gaussian, dim=-1)
        self.splats['means'].grad = None
        self.splats['quats'].grad = None
        self.splats['scales'].grad = None
        self.splats['opacities'].grad = None
        self.splats['sh0'].grad = None
        self.splats['shN'].grad = None
        multi_certainties = []
        for exp_index in self.c_exp_index:
            inv_H_gaussian = torch.exp(-exp_index * H_per_gaussian)
            certainties, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                override_color=inv_H_gaussian,
                sh_degree=None,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
            )  # [1, H, W, 3]
            certainties = certainties[0].detach()
            certainties = reduce(certainties, "h w c -> h w", "mean").detach()
            certainties = (alphas * certainties).clamp(0, 1)
            certainties = soft_sigmoid(certainties - 0.5, soft=10.0)
            multi_certainties.append(certainties)
        return colors, multi_certainties, alphas, depths

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, render_alphas, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            # skip GSs that have small image radius (in pixels)
            radius_clip=3.0,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy(), render_alphas.squeeze().cpu().numpy()

    def render(self, idx, split="test", eval=False):
        device = self.device
        if split == "test":
            data = self.test_dataset[idx]
        elif split == "train":
            data = self.train_dataset[idx]
        else:
            raise ValueError
        c2w = data["camtoworld"].float()
        c2w = c2w[None, ...].to(device)
        Ks = data["K"][None, ...].to(device)
        if data["image"] is not None:
            height, width = data["image"].shape[:2]
        else:
            cam_id = 1 if split == "test" else 0
            width, height = self.parser.imsize_dict[cam_id]
        colors, multi_certainties, alphas, depths = self.rasterize_splats_w_certainty(
            camtoworlds=c2w,
            Ks=Ks,
            width=width,
            height=height,
        )
        cam_param = {
            "c2w": c2w[0],
            "K": Ks[0],
        }

        eval_results = None
        if eval and data["image"] is not None:
            psnr = self.psnr(colors, data["image"].to(device) / 255.0).item()
            ssim = self.ssim(colors.permute(2, 0, 1)[None, ...], data["image"].permute(
                2, 0, 1)[None, ...].to(device) / 255.0).item()
            lpips = self.lpips(colors.permute(2, 0, 1)[None, ...], data["image"].permute(
                2, 0, 1)[None, ...].to(device) / 255.0).item()
            eval_results = {
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips,
            }

        # #debug
        # torch.save(colors.detach().cpu(), f"dbg/evaluation/renders_{idx}.pt")
        # torch.save(data["image"] / 255. , f"dbg/evaluation/gts_{idx}.pt")

        return colors, multi_certainties, alphas, depths, cam_param, eval_results

    def refine(self, refine_cams, train_cams, train_prob, max_steps=100, gen_loss_weight=0.2, use_affine=True):
        cfg = self.cfg
        device = self.device
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # Training loop.
        densification = True
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:

            if step <= max_steps*1/3:
                is_refine_step = step % 3 == 1
            elif step <= max_steps*2/3:
                is_refine_step = step % 5 == 1
            else:
                is_refine_step = step % 8 == 1

            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            # get data
            affine = None
            if is_refine_step:
                idx = np.random.randint(0, len(refine_cams))
                data = refine_cams[idx]
            else:
                # idx = np.random.randint(0, len(train_cams))
                normalized_prob = train_prob / np.sum(train_prob)
                idx = np.random.choice(
                    len(train_cams), 1, p=normalized_prob).item()
                data = train_cams[idx]
            if data.get('Gen', False):
                affine = self.affines[data['image_id']]

            camtoworlds = data["camtoworld"][None, ...].to(device)  # [1, 4, 4]
            Ks = data["K"][None, ...].to(device)  # [1, 3, 3]
            pixels = data["image"][None, ...].to(
                device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )

            height, width = pixels.shape[1:3]

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
                affine=affine if use_affine else None,
            )
            colors = renders.clip(0, 1)

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if densification:
                self.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # loss
            ssimloss = 1.0 - self.ssim(
                pixels.permute(0, 3, 1, 2), colors.permute(0, 3, 1, 2)
            )
            if data.get('Gen', False):
                # loss = l1loss * 0.2 + ssimloss * 0.4 + lpips_loss * 0.4
                l1loss = F.l1_loss(colors, pixels)
                # l1loss = neighbor_L1_loss(colors, pixels)
                # loss = ssimloss * 0.1 + l1loss * 0.2
                loss = l1loss * gen_loss_weight
            else:
                l1loss = F.l1_loss(colors, pixels)
                loss = l1loss * (1.0 - cfg.ssim_lambda) + \
                    ssimloss * cfg.ssim_lambda

            loss.backward()

            if densification:
                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # logging
            desc = f"loss={loss.item():.3f}"
            pbar.set_description(desc)
            if cfg.tb_every > 0 and self.total_step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar(
                    "train/loss", loss.item(), self.total_step)
                self.writer.add_scalar(
                    "train/l1loss", l1loss.item(), self.total_step)
                self.writer.add_scalar(
                    "train/ssimloss", ssimloss.item(), self.total_step)
                self.writer.add_scalar(
                    "train/num_GS", len(self.splats["means"]), self.total_step)
                self.writer.add_scalar("train/mem", mem, self.total_step)
                if cfg.tb_save_image:
                    canvas = torch.cat(
                        [pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image(
                        "train/render", canvas, self.total_step)
                self.writer.flush()

            self.total_step += 1

            # optimize
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if affine is not None:
                self.affine_optimizers[f"{data['image_id']}"].step()
                self.affine_optimizers[f"{data['image_id']}"].zero_grad(
                    set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            # update viewer
            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(self.total_step, num_train_rays_per_step)

    def save(self):
        # Save checkpoint
        mean_numpy = self.splats["means"].detach(
        ).cpu().numpy().astype(np.float32)
        scales_numpy = self.splats["scales"].detach(
        ).cpu().numpy().astype(np.float32)
        opacities_numpy = self.splats["opacities"].unsqueeze(
            -1).detach().cpu().numpy().astype(np.float32)
        quats_numpy = self.splats["quats"].detach(
        ).cpu().numpy().astype(np.float32)
        sh0_numpy = self.splats["sh0"].permute(
            0, 2, 1).detach().cpu().numpy().astype(np.float32)
        # [N, 3, 1]
        shN_numpy = self.splats["shN"].permute(
            0, 2, 1).detach().cpu().numpy().astype(np.float32)
        # [N, 3, S-1]

        _write_splat_data_to_ply(
            Path(f"{self.cfg.result_dir}/splat.ply"),
            xyz=mean_numpy,
            scales=scales_numpy,
            opacities=opacities_numpy,
            rots=quats_numpy,
            features_dc=sh0_numpy,
            features_extra=shN_numpy,
        )

    @torch.no_grad()
    def render_refined_video(self):
        pass
        # frames = []
        # for idx in range(len(self.refine_dataset.all_image_paths)):
        #     camtoworlds = torch.from_numpy(self.refine_dataset.all_interp_c2ws[idx]).float()[None,...].to(self.device)
        #     Ks = torch.from_numpy(self.refine_dataset.closest_K).float()[None,...].to(self.device)
        #     img_wh = self.refine_dataset.img_wh

        #     colors, alphas, info = self.rasterize_splats(
        #         camtoworlds=camtoworlds,
        #         Ks=Ks,
        #         width=img_wh[0],
        #         height=img_wh[1],
        #         sh_degree=self.cfg.sh_degree,
        #         near_plane=self.cfg.near_plane,
        #         far_plane=self.cfg.far_plane,
        #         image_ids=idx,
        #         render_mode="RGB",
        #     )
        #     frames.append(np.clip(colors[0].cpu().numpy(),0,1))

        # save_dir = f"{self.cfg.result_dir}/to_refine"
        # writer = imageio.get_writer(f"{save_dir}/refined_gs_render.mp4", fps=6)
        # for frame in frames:
        #     writer.append_data((frame*255).astype(np.uint8))
        # writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, help="the path to the config file", default='results/bike/cfg.json')
    args = parser.parse_args()

    with open(args.cfg, "r") as f:
        config = Config(**json.load(f))

    refiner = Refiner(config)
    refiner.train()

    if not config.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
