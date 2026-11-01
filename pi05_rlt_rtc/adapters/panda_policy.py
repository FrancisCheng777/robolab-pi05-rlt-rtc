"""Real Panda dataset transforms and a dedicated RLinf/OpenPI configuration.

The public OpenPI DataConfigFactory interface is used; no simulator policy,
velocity scaling, or extra DeltaActions/AbsoluteActions transform is involved.
"""
from __future__ import annotations

import dataclasses
import numpy as np

from pi05_rlt_rtc.contracts.errors import ContractError
from pi05_rlt_rtc.contracts.physical import MODEL_CONFIG, PROMPT


def rgb_image(value):
    value = np.asarray(value)
    if value.ndim != 3:
        raise ContractError("camera image must be RGB HWC or CHW")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = value.transpose(1, 2, 0)
    if value.shape[-1] != 3:
        raise ContractError("camera image must have three channels")
    if np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ContractError("floating RGB must be finite and in [0,1]")
        value = np.rint(value * 255).astype(np.uint8)
    if value.dtype != np.uint8:
        raise ContractError("RGB must be uint8 or normalized float")
    return value


@dataclasses.dataclass(frozen=True)
class PandaInputs:
    default_prompt: str = PROMPT

    def __call__(self, data):
        state = np.asarray(data["observation/state"], np.float32)
        if state.shape != (9,) or not np.isfinite(state).all():
            raise ContractError("physical state requires q[7] + two measured aperture/2 values")
        from pi05_rlt_rtc.data.profile import model_images
        # RLinf names its first camera observation/image, regardless of mounting.
        wrist = rgb_image(data["observation/wrist_image"] if "observation/wrist_image" in data else data["observation/image"])
        images, masks = model_images(wrist)
        out = {"state": state, "image": images, "image_mask": masks,
               "prompt": data.get("prompt", data.get("task", self.default_prompt))}
        if "actions" in data:
            actions = np.asarray(data["actions"], np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 8 or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
                raise ContractError("SFT actions require finite normalized [H,8] physical commands")
            out["actions"] = actions
        return out


@dataclasses.dataclass(frozen=True)
class PandaOutputs:
    def __call__(self, data):
        actions = np.asarray(data["actions"])
        return {"actions": actions[..., :8]}


def physical_norm_stats(stats):
    """Match the collection project's FP32 quantiles before state tokenization.

    JSON-loaded FP64 quantiles can change normalized near-constant joints enough
    to cross a discrete-state bin. SFT and inference must use the same precision.
    """
    if stats is None:
        return None
    return {key: dataclasses.replace(value, **{
        field.name: np.asarray(getattr(value, field.name), dtype=np.float32)
        for field in dataclasses.fields(value) if getattr(value, field.name) is not None
    }) for key, value in stats.items()}


def panda_train_config(tokenizer_path=None, max_token_len=200, action_horizon=10):
    """Construct lazily so controller-only installations need no Torch/OpenPI."""
    import openpi.transforms as transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.training.config import AssetsConfig, DataConfig, DataConfigFactory, ModelTransformFactory, TrainConfig

    @dataclasses.dataclass(frozen=True)
    class PandaDataConfig(DataConfigFactory):
        default_prompt: str = PROMPT

        def create(self, assets_dirs, model_config):
            base = self.create_base_config(assets_dirs, model_config)
            if tokenizer_path is None:
                model_transforms = ModelTransformFactory()(model_config)
            else:
                from .tokenizer import LocalPaliGemmaTokenizer
                model_transforms = transforms.Group(inputs=[
                    transforms.InjectDefaultPrompt(self.default_prompt),
                    transforms.ResizeImages(224, 224),
                    transforms.TokenizePrompt(LocalPaliGemmaTokenizer(tokenizer_path, max_token_len),
                                              discrete_state_input=True),
                    transforms.PadStatesAndActions(32)])
            return dataclasses.replace(base, norm_stats=physical_norm_stats(base.norm_stats),
                repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
                    "observation/wrist_image": "observation.images.wrist",
                    "observation/state": "observation.state", "actions": "action"})]),
                data_transforms=transforms.Group(inputs=[PandaInputs(self.default_prompt)], outputs=[PandaOutputs()]),
                model_transforms=model_transforms, action_sequence_keys=("action",))

    return TrainConfig(name=MODEL_CONFIG,
        model=Pi0Config(pi05=True, action_horizon=action_horizon, max_token_len=max_token_len, discrete_state_input=True),
        data=PandaDataConfig(repo_id="local/pi05_plug_train",
            base_config=DataConfig(prompt_from_task=False), assets=AssetsConfig()),
        pytorch_weight_path="/configure/pi05_base", batch_size=32, num_workers=2)


def register_panda_profile(tokenizer_path=None, max_token_len=200, action_horizon=10):
    from rlinf.models.embodiment.openpi import dataconfig
    if tokenizer_path is not None or MODEL_CONFIG not in dataconfig._CONFIGS_DICT:
        dataconfig._CONFIGS_DICT[MODEL_CONFIG] = panda_train_config(tokenizer_path, max_token_len, action_horizon)
