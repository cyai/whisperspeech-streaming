from WhisperSpeech.whisperspeech.modules import *
from WhisperSpeech.whisperspeech.a2wav import Vocoder
from WhisperSpeech.whisperspeech.pipeline import Pipeline
from WhisperSpeech.whisperspeech import languages, inference
from WhisperSpeech.whisperspeech.s2a_delar_mup_wds_mlang import (
    SADelARTransformer as SADelARTransformerBase,
)
from WhisperSpeech.whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer
from WhisperSpeech.whisperspeech.s2a_delar_mup_wds_mlang_cond import SADelARTransformer

import torch
import traceback
from pathlib import Path
import torch.nn.functional as F
from fastprogress import progress_bar
from torch.profiler import record_function


class StreamingPipeline(Pipeline):
    def __init__(
        self,
        t2s_ref=None,
        s2a_ref=None,
        optimize=True,
        torch_compile=False,
        device=None,
    ):
        if device is None:
            device = inference.get_compute_device()
        self.device = device
        args = dict(device=device)
        try:
            if t2s_ref:
                args["ref"] = t2s_ref
            self.t2s = StreamingTSARTransformer.load_model(
                **args
            )  # use obtained compute device
            # self.t2s = TSARTransformer.load_model(**args)  # use obtained compute device
            if optimize:
                self.t2s.optimize(torch_compile=torch_compile)
        except:
            print("Failed to load the T2S model:")
            print(traceback.format_exc())
        args = dict(device=device)
        try:
            if s2a_ref:
                spec = inference.load_model(ref=s2a_ref, device=device)
                if [
                    x
                    for x in spec["state_dict"].keys()
                    if x.startswith("cond_embeddings.")
                ]:
                    cls = StreamingSADelARTransformer
                    # cls = s2a_delar_mup_wds_mlang_cond.SADelARTransformer
                    args["spec"] = spec
                else:
                    cls = StreamingSADelARTransformerBase
                    args["spec"] = spec
            else:
                cls = StreamingSADelARTransformerBase
            self.s2a = cls.load_model(**args)  # use obtained compute device
            if optimize:
                self.s2a.optimize(torch_compile=torch_compile)
        except:
            print("Failed to load the S2A model:")
            print(traceback.format_exc())

        self.vocoder = Vocoder(device=device)
        self.encoder = None

    def generate_atoks(self, text, speaker=None, lang="en", cps=15, step_callback=None):
        if speaker is None:
            speaker = self.default_speaker
        elif isinstance(speaker, (str, Path)):
            speaker = self.extract_spk_emb(speaker)
        text = text.replace("\n", " ")
        stoks = self.t2s.generate(text, cps=cps, lang=lang, step=step_callback)[0]
        atoks = self.s2a.generate(stoks, speaker.unsqueeze(0), step=step_callback)
        yield atoks

    def generate(self, text, speaker=None, lang="en", cps=15, step_callback=None):
        yield self.vocoder.decode(
            self.generate_atoks(
                text, speaker, lang=lang, cps=cps, step_callback=step_callback
            )
        )


class StreamingSADelARTransformerBase(SADelARTransformerBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        stoks,
        speakers,
        langs=None,
        atoks_prompt=None,
        N=None,
        bs=1,
        T=0.7,
        top_k=None,
        show_progress_bar=True,
        step=None,
        subsample_enc=False,
    ):
        dev = self.device
        N = N or len(stoks) * 3
        stoks = F.pad(
            stoks.to(dev),
            (1, self.stoks_len - len(stoks) - 1),
            value=self.stoks_codes - 1,
        ).unsqueeze(0)
        speakers = speakers.to(device=dev, dtype=self.dtype)
        toks = torch.full(
            (bs, self.quantizers, self.ctx_n),
            self.codes + 1,
            dtype=torch.long,
            device=dev,
        )
        T = torch.tensor(T, device=dev)

        start = 0  # number of valid tokens or the index of first empty spot
        if atoks_prompt is not None:
            start = atoks_prompt.shape[-1]
            for i in range(self.quantizers):
                toks[:, i, 1 + i : start + i + 1] = atoks_prompt[:, i]
        start += 1  # we always start with at least an SOT

        with record_function("encode"):
            stoks, speakers = [x.repeat(bs, 1) for x in (stoks, speakers)]
            xenc, xenc_positions, _ = self.run_encoder(stoks, speakers)
            toks_positions = torch.arange(N, device=dev)
        with record_function("prefill"):
            initial = self.generate_one(
                toks[:, :, :start],
                toks_positions[:start],
                langs,
                xenc,
                xenc_positions,
                T,
                top_k,
            )
            toks[:, :start, start : start + 1] = initial[:, :start]
            start += 1

        with inference.inference_context():
            it = range(start, min(N, self.ctx_n - 1))
            if show_progress_bar:
                it = progress_bar(it)

            for i in it:
                with record_function("generate_one"):
                    toks[:, :i, i : i + 1] = self.generate_next(
                        toks[:, :, i - 1 : i],
                        toks_positions[i - 1 : i],
                        langs,
                        xenc,
                        xenc_positions,
                        T,
                        top_k,
                    )[:, :i]

                    yield toks[:, :, i : i + 1]

                # for profiling, debugging or early exit
                if step is not None:
                    step()
        # shift tokens
        toks = toks[:, :, 1:N]
        for j in range(self.quantizers):
            toks[:, j] = torch.roll(toks[:, j], -j)
        # return toks[:, :, : N - 4]
        yield toks[:, :, : N - 4]


class StreamingTSARTransformer(TSARTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        txt,
        cps=15,
        lang="en",
        stoks_prompt=None,
        N=None,
        bs=1,
        T=0.7,
        top_k=None,
        step=None,
        show_progress_bar=True,
    ):
        self.ensure_tokenizer()
        N = N or self.stoks_len
        dev = self.device
        ttoks = []
        langs = []
        if isinstance(lang, list):
            lang0 = lang[0]
            assert isinstance(
                txt, list
            ), "lang and txt have to be both lists or strings"
            for txt, lang in zip(txt, lang):
                tt = self.tokenizer.encode(txt)
                ttoks += tt
                langs += [languages.to_id(lang)] * len(tt)
        elif isinstance(lang, torch.Tensor):
            langs = lang
            ttoks = self.tokenizer.encode(txt)
        else:
            lang0 = lang
            ttoks = self.tokenizer.encode(txt)
            langs = torch.tensor([languages.to_id(lang)], device=dev)
        ttoks = torch.tensor(ttoks, device=dev)
        ttoks = F.pad(
            ttoks, (1, self.ttoks_len - len(ttoks) - 1), value=self.tokenizer.eot
        )
        cpss = torch.tensor([cps], device=dev)
        T = torch.tensor(T, device=dev)
        if not isinstance(langs, torch.Tensor):
            langs = torch.tensor(langs, device=dev)
            langs = F.pad(
                langs,
                (1, self.ttoks_len - len(langs) - 1),
                value=languages.to_id(lang0),
            )

        toks = torch.zeros((bs, N), dtype=torch.long, device=dev)
        toks[:, 0] = self.stoks_codes + self.tunables.padding_token_offset
        start = 0
        if stoks_prompt is not None:
            toks[:, 1 : len(stoks_prompt) + 1] = stoks_prompt
            start = len(stoks_prompt)
        it = range(start + 1, N - 1)
        if show_progress_bar:
            it = progress_bar(it)

        toks_positions = torch.arange(N, device=dev)
        with record_function("encode"):
            ttoks = ttoks.repeat(bs, 1)
            langs, cpss = [x.repeat(bs) for x in (langs, cpss)]
            xenc, xenc_positions, cps_emb = self.run_encoder(ttoks, langs, cpss)
            toks_positions = torch.arange(N + 1, device=dev)

        with record_function("prefill"):
            toks[:, start + 1] = self.generate_one(
                toks[:, : start + 1].contiguous(),
                toks_positions[: start + 1],
                cps_emb,
                xenc,
                xenc_positions,
                T,
                top_k,
            )[:, 0]
        with inference.inference_context():
            for i in it:
                toks[:, i + 1] = self.generate_next(
                    toks[:, i : i + 1],
                    toks_positions[i : i + 1],
                    cps_emb,
                    xenc,
                    xenc_positions,
                    T,
                    top_k,
                )[:, 0]
                yield toks[:, i + 1]

                if (
                    toks[:, i + 1]
                    == self.stoks_codes + self.tunables.padding_token_offset
                ).all():
                    yield toks[:, 1 : i + 1]

                # for profiling, debugging or early exit
                if step is not None:
                    step()
        # return toks[:, 1:]


class StreamingSADelARTransformer(SADelARTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        stoks,
        speakers,
        langs=None,
        atoks_prompt=None,
        N=None,
        bs=1,
        T=0.7,
        top_k=None,
        show_progress_bar=True,
        step=None,
        subsample_enc=False,
    ):
        dev = self.device
        N = N or len(stoks) * 3
        stoks = F.pad(
            stoks.to(dev),
            (1, self.stoks_len - len(stoks) - 1),
            value=self.stoks_codes - 1,
        ).unsqueeze(0)
        speakers = speakers.to(device=dev, dtype=self.dtype)
        toks = torch.full(
            (bs, self.quantizers, self.ctx_n),
            self.codes + 1,
            dtype=torch.long,
            device=dev,
        )
        T = torch.tensor(T, device=dev)

        start = 0  # number of valid tokens or the index of first empty spot
        if atoks_prompt is not None:
            start = atoks_prompt.shape[-1]
            for i in range(self.quantizers):
                toks[:, i, 1 + i : start + i + 1] = atoks_prompt[:, i]
        start += 1  # we always start with at least an SOT

        with record_function("encode"):
            stoks, speakers = [x.repeat(bs, 1) for x in (stoks, speakers)]
            xenc, xenc_positions, _ = self.run_encoder(
                stoks, [dict(speaker=s, snr=60, c50=60) for s in speakers]
            )
            toks_positions = torch.arange(N, device=dev)
        with record_function("prefill"):
            initial = self.generate_one(
                toks[:, :, :start],
                toks_positions[:start],
                langs,
                xenc,
                xenc_positions,
                T,
                top_k,
            )
            toks[:, :start, start : start + 1] = initial[:, :start]
            start += 1

        with inference.inference_context():
            it = range(start, min(N, self.ctx_n - 1))
            if show_progress_bar:
                it = progress_bar(it)

            for i in it:
                with record_function("generate_one"):
                    toks[:, :i, i : i + 1] = self.generate_next(
                        toks[:, :, i - 1 : i],
                        toks_positions[i - 1 : i],
                        langs,
                        xenc,
                        xenc_positions,
                        T,
                        top_k,
                    )[:, :i]

                    yield toks[:, :, i : i + 1]

                    # yield torch.roll(toks[:, :, i : i + 1], -i, 1)
                # for profiling, debugging or early exit
                if step is not None:
                    step()
        # shift tokens
        toks = toks[:, :, 1:N]
        for j in range(self.quantizers):
            toks[:, j] = torch.roll(toks[:, j], -j)
        yield toks[:, :, : N - 4]
