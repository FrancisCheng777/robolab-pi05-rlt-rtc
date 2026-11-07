"""Local SentencePiece tokenizer with exactly the Stage-1 Pi0.5 prompt format."""
import numpy as np
from pi05_rlt_rtc.contracts.errors import ContractError


class LocalPaliGemmaTokenizer:
    def __init__(self, path, max_len=200):
        import sentencepiece
        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.max_len = max_len

    def tokenize(self, prompt, state=None):
        if state is None or np.asarray(state).shape != (9,) or not np.isfinite(state).all():
            raise ContractError('physical tokenizer requires nine normalized state values before padding')
        cleaned = prompt.strip().replace('_', ' ').replace('\n', ' ')
        bins = np.digitize(state, np.linspace(-1., 1., 257)[:-1]) - 1
        text = f"Task: {cleaned}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self.processor.encode(text, add_bos=True)
        if len(tokens) > self.max_len:
            raise ContractError('Pi0.5 prompt would truncate; match a larger Stage-1 max_token_len')
        return (np.asarray(tokens + [0] * (self.max_len - len(tokens)), np.int64),
                np.arange(self.max_len) < len(tokens))
