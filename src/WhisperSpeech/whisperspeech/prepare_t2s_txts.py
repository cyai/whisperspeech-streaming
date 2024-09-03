# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/3A. T2S transcripts preparation.ipynb.

# %% auto 0
__all__ = []

# %% ../nbs/3A. T2S transcripts preparation.ipynb 2
import sys
import os
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fastprogress import progress_bar
from fastcore.script import *

import whisper, whisperx
from . import utils, vad_merge
import webdataset as wds

from .inference import get_compute_device

# %% ../nbs/3A. T2S transcripts preparation.ipynb 4
class Transcriber:
    """
    A helper class to transcribe a batch of 30 second audio chunks.
    """
    def __init__(self, model_size, lang=False):
        self.model_size = model_size
        # try to translate long language names to codes
        lang = whisper.tokenizer.TO_LANGUAGE_CODE.get(lang, lang)
        self.model = whisperx.asr.load_model(
            model_size, get_compute_device(), compute_type="float16", language=lang,
            asr_options=dict(repetition_penalty=1, no_repeat_ngram_size=0, prompt_reset_on_temperature=0.5,
                             max_new_tokens=500, clip_timestamps=None, hallucination_silence_threshold=None))
        # without calling vad_model at least once the rest segfaults for some reason...
        self.model.vad_model({"waveform": torch.zeros(1, 16000), "sample_rate": 16000})
        
    def transcribe(self, batch):
        batch = whisper.log_mel_spectrogram(batch, 128 if self.model_size == 'large-v3' else 80)
        embs = self.model.model.encode(batch.cpu().numpy())
        return self.model.tokenizer.tokenizer.decode_batch([x.sequences_ids[0] for x in 
            self.model.model.model.generate(
                embs,
                [self.model.model.get_prompt(self.model.tokenizer, [], without_timestamps=True)]*len(batch),
            )])

# %% ../nbs/3A. T2S transcripts preparation.ipynb 5
@call_parse
def prepare_txt(
    input:str,           # input shard URL/path
    output:str,          # output shard path
    n_samples:int=None, # process a limited amount of samples
    batch_size:int=16, # process several segments at once
    transcription_model:str="medium",
    language:str="en",
):
    transcriber = Transcriber(transcription_model, lang=language)

    total = n_samples//batch_size if n_samples else 'noinfer'
    if n_samples: print(f"Benchmarking run of {n_samples} samples ({total} batches)")

    import math, time
    start = time.time()
    ds = wds.WebDataset([utils.derived_name(input, 'mvad')]).decode()
    total = math.ceil(sum([len(x['raw.spk_emb.npy']) for x in ds])/batch_size)
    print(f"Counting {total} batches: {time.time()-start:.2f}")

    ds = vad_merge.chunked_audio_dataset([input], 'raw').compose(
        utils.resampler(16000, 'samples_16k'),
    )

    ds = ds.compose(
        wds.to_tuple('__key__', 'rpad', 'samples_16k'),
        wds.batched(64),
    )

    dl = wds.WebLoader(ds, num_workers=1, batch_size=None).unbatched().batched(batch_size)

    with utils.AtomicTarWriter(output, throwaway=n_samples is not None) as sink:
        for keys, rpads, samples in progress_bar(dl, total=total):
            csamples = samples.to(get_compute_device())
            txts = transcriber.transcribe(csamples)

            for key, rpad, txt in zip(keys, rpads, txts):
                sink.write({
                    "__key__": key,
                    "txt": txt,
                })
