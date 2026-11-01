# 固定 RLinf 源码接口检查

2026-10-29；提交 `c70606f08cdca259b8dec03d4430926b5b8fac9d`。13/13 项源码检查通过。

这是静态接口检查，不等于真实模型执行。物理 profile 的两处显式注册已在隔离 checkout 应用；没有修改远端正在开发的训练副本。

| 检查 | 状态 | 证据 |
|---|---|---|
| C03.extract_rlt_obs_plain_euler | CONFIRMED | L122: def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]: |
| C03.rtc_only_approx | CONFIRMED | L81: f"Unsupported RTC guidance mode: {self.rtc_guidance_mode!r}" |
| C03.full_horizon_decode_truncated | CONFIRMED | L87: def output_transform(self, outputs: dict) -> dict: |
| C06.output_transform_action_chunk_truncation | CONFIRMED | L109: recombined["actions"] = recombined["actions"][:, :action_chunk] |
| C04.approx_only | CONFIRMED | L50: def build_rtc_target_and_mask( |
| C08.ref_chunk_first_C | CONFIRMED | L104: ref_chunk = ref_chunk[:, : self.chunk_len] |
| C08.sft_forward_not_tanh | CONFIRMED | L204: return F.mse_loss(pred_actions, target_actions, reduction="none") |
| C09.learner_ref_chunk_first_C | CONFIRMED | L69: return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1) |
| C09.critic_discount_uses_tensor_width | CONFIRMED | L87: def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor: |
| C09.rlt_ac_loss_recipe | CONFIRMED | L355: actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss |
| C02.rlt_alpha_multiplies_vla_loss | CONFIRMED | L844: "loss": rlt_loss + self.rlt_cfg.rlt_alpha * vla_loss, |
| C11.checkpoint_rlt_completeness | CONFIRMED | L96: extra heads (``rlt_module.*``, ``value_head.*``, …) stay as-is. |
| C15.rlt_ac_no_training_pipeline | CONFIRMED | L62: "runner.use_training_pipeline=True is not supported for rlt_ac." |

完整记录：[source_audit.json](../runs/physical_migration/source_audit.json)。本地依赖：[environment.json](../runs/physical_migration/environment.json)。真实 OpenPI/LeRobot/GPU 端到端验证仍待执行。
