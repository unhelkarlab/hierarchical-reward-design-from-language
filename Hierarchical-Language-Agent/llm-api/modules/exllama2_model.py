import torch
import time

from exllamav2 import (ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer,
                       ExLlamaV2Cache)
import exllamav2.generator
from exllamav2.generator import (ExLlamaV2BaseGenerator,
                                 ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob,
                                 ExLlamaV2Sampler)

import modules.shared as shared
from modules.callbacks import Iteratorize


class ExLlamaV2Model:

  def __init__(self) -> None:
    pass

  @classmethod
  def from_pretrained(self, path):

    def init(path,
             args,
             quiet: bool = False,
             allow_auto_split: bool = False,
             skip_load: bool = False,
             benchmark: bool = False,
             max_batch_size: int = None,
             max_input_len: int = None,
             max_output_len: int = None,
             progress: bool = False):

      # Create config

      config = ExLlamaV2Config()
      config.model_dir = 'models/' + path
      config.fasttensors = hasattr(args,
                                   "fast_safetensors") and args.fast_safetensors
      config.prepare()

      # Set config options

      if args.length: config.max_seq_len = args.length
      if args.rope_scale: config.scale_pos_emb = args.rope_scale
      if args.rope_alpha: config.scale_alpha_value = args.rope_alpha
      config.no_flash_attn = args.no_flash_attn
      config.no_xformers = args.no_xformers
      if args.experts_per_token:
        config.num_experts_per_token = args.experts_per_token

      if max_batch_size: config.max_batch_size = max_batch_size
      config.max_output_len = max_output_len
      if max_input_len: config.max_input_len = max_input_len

      # Set low-mem options

      if args.low_mem: config.set_low_mem()
      if args.load_q4: config.load_in_q4 = True

      # Load model
      # If --gpu_split auto, return unloaded model. Model must be loaded with
      # model.load_autosplit() supplying cache created in lazy mode

      model = ExLlamaV2(config)

      split = None
      if args.gpu_split and args.gpu_split != "auto":
        split = [float(alloc) for alloc in args.gpu_split.split(",")]

      if args.gpu_split != "auto" and not skip_load:
        if not quiet and not progress: print(" -- Loading model...")
        t = time.time()
        #   model.load(split, progress=progress)
        model.load(split)
        t = time.time() - t
        if benchmark and not quiet:
          print(f" -- Loaded model in {t:.4f} seconds")
      else:
        assert allow_auto_split, "Auto split not allowed."

      # Load tokenizer

      if not quiet: print(" -- Loading tokenizer...")

      tokenizer = ExLlamaV2Tokenizer(config)

      return model, tokenizer

    result = self()
    result.model, result.tokenizer = init(
        path,
        shared.args,
        skip_load=shared.args.stream_layers,
        benchmark=True,
        max_output_len=shared.args.max_output_len)
    result.cache = ExLlamaV2Cache(result.model)
    result.generator = ExLlamaV2DynamicGenerator(result.model, result.cache,
                                                 result.tokenizer)
    if not shared.args.no_warmup: result.generator.warmup()
    return result, result

  def encode(self, string):
    return self.tokenizer.encode(string)

  def decode(self, tokens):
    return self.tokenizer.decode(tokens)

  def generate(self, prompt, state, callback=None):
    with torch.inference_mode():

      print(f" -- Generating...")
      # print('Prompt: ', prompt)

      settings = ExLlamaV2Sampler.Settings()
      # print('State: ', state)
      settings.temperature = state['temperature']
      settings.top_k = state['top_k']
      settings.top_p = state['top_p']
      settings.token_repetition_penalty = state['repetition_penalty']

      time_begin = time.time()

      job = ExLlamaV2DynamicJob(input_ids=self.tokenizer.encode(prompt,
                                                                add_bos=True),
                                max_new_tokens=state['max_new_tokens'],
                                stop_conditions=[self.tokenizer.eos_token_id],
                                gen_settings=settings)
      self.generator.enqueue(job)

      output = ""
      while self.generator.num_remaining_jobs():
        results = self.generator.iterate()

        # iterate() always returns a list of zero or more result dicts
        for result in results:
          # The text key will only be present during the streaming stage and
          # may be an empty string
          text_chunk = result.get("text", "")

          # print('A text chunk: ' + text_chunk, end="")

          # Collect all the outputs
          output += text_chunk

          if callback:
            callback(text_chunk)

      torch.cuda.synchronize()

      time_end = time.time()

    print('Output: ', output)
    print()

    total_gen = time_end - time_begin
    print(f" -- Response generated in {total_gen:.2f} seconds.")

    return output

  def generate_with_streaming(self, *args, **kwargs):
    with Iteratorize(self.generate, args, kwargs, callback=None) as generator:
      reply = ''
      for token in generator:
        reply += token
        print('Output: ', reply)
        yield reply
