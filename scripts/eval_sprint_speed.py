"""eval_sprint_speed.py — 在 warp 训练环境内直接测 checkpoint 的真实速度。

绕开本地 CPU harness 的所有差异项（BAM/延迟/噪声/归一化烘焙），用训练同款
环境 + rsl_rl 推理策略（含 obs normalizer）回答：策略在原生环境里跑多快？

用法（实例上，工作目录 = repo 根）：
    uv run --no-sync python scripts/eval_sprint_speed.py \
        --checkpoint logs/rsl_rl/sprint/<run>/model_1999.pt --vx 0.4 1.2 2.0
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Mjlab-Sprint-Flat-MicroDuck")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--onnx", default=None, help="测 ONNX（归一化已烘焙）而非 .pt checkpoint")
    ap.add_argument("--vx", type=float, nargs="+", default=[0.4, 0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import mjlab.tasks  # noqa: F401  — populate the registry
    import mjlab_microduck.tasks  # noqa: F401  — microduck registrations
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper, MjlabOnPolicyRunner

    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args.onnx:
        import onnxruntime as ort
        session = ort.InferenceSession(args.onnx)

        def policy(obs):
            vec = obs["actor"].detach().cpu().numpy().astype(np.float32)
            # 导出的 ONNX batch 维固定为 1，逐 env 推理
            acts = [session.run(None, {session.get_inputs()[0].name: vec[i : i + 1]})[0]
                    for i in range(vec.shape[0])]
            return torch.tensor(np.concatenate(acts, axis=0),
                                device=args.device, dtype=torch.float32)
    else:
        assert args.checkpoint, "--checkpoint or --onnx required"
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True,
                    map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

    uw = env.unwrapped
    robot = uw.scene["robot"]
    cmd_term = uw.command_manager.get_term("twist")
    dt = uw.step_dt

    print(f"policy: {args.onnx or args.checkpoint}")
    print(f"control dt: {dt:.4f}s  envs: {args.num_envs}  task: {args.task}")

    def force_cmd(vx):
        cmd_term.vel_command_b[:, 0] = vx
        cmd_term.vel_command_b[:, 1] = 0.0
        cmd_term.vel_command_b[:, 2] = 0.0
        cmd_term.is_standing_env[:] = False

    obs, _ = env.reset()
    for vx in args.vx:
        obs, _ = env.reset()
        force_cmd(vx)
        # 2s 加速/稳定段
        for _ in range(int(2.0 / dt)):
            force_cmd(vx)
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        x0 = robot.data.root_link_pos_w[:, 0].clone()
        y0 = robot.data.root_link_pos_w[:, 1].clone()
        n_meas = int(args.seconds / dt)
        dones_total = 0
        for _ in range(n_meas):
            force_cmd(vx)
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            dones_total += int(dones.sum())
        x1 = robot.data.root_link_pos_w[:, 0]
        y1 = robot.data.root_link_pos_w[:, 1]
        dist = torch.hypot(x1 - x0, y1 - y0)
        speed = dist / args.seconds
        alive = (dones_total == 0)
        print(f"cmd={vx:4.1f}  实测 speed mean={speed.mean():.3f}  "
              f"p10={speed.quantile(0.1):.3f}  p90={speed.quantile(0.9):.3f}  "
              f"max={speed.max():.3f} m/s  resets={dones_total}")

    env.close()


if __name__ == "__main__":
    main()
