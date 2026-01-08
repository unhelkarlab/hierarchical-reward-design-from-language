import requests as rq
from openai import OpenAI
import concurrent.futures
import time
import threading
import html
from typing import List


def with_retry(func, times=5, interval=1):

  def wrapper(*args, **kwargs):
    for _ in range(times):
      try:
        return func(*args, **kwargs)
      except Exception as e:
        print(e)
        print("Retrying...")
        time.sleep(interval)

  return wrapper


class LLM:

  def __call__(self, chatter: callable):
    raise NotImplementedError

  def eval_prob(self, prompt: List[List[str]], choice: List[str]):
    raise NotImplementedError


class LLM_GPT_API(LLM):
  MAX_RETRY_TIMES = 5

  def __init__(self,
               models: List[str],
               api_key: str,
               api_org: str,
               temp: float = 0.01,
               stream: bool = False,
               max_retry: int = 10):
    self.models = models
    self.client = OpenAI(api_key=api_key, organization=api_org)
    self.stream = stream
    self.temp = temp
    self.n = 1
    self.num_retry = 0
    self.max_retry = max_retry

    # concurrency control
    self.lock = threading.Lock()
    self.model_idx = 0

  @with_retry
  def _infer(self, msg, timeout=2.5) -> str:
    with self.lock:
      model = self.models[self.model_idx]
      self.model_idx = (self.model_idx + 1) % len(self.models)

    if self.stream:
      response = self.client.chat.completions.create(
          model=model,
          temperature=self.temp,
          messages=msg,
          stream=True  # again, we set stream=True
      )
      # create variables to collect the stream of chunks
      collected_messages = []
      # iterate through the stream of events
      for chunk in response:
        # extract the message
        chunk_message = chunk.choices[0].delta.content
        collected_messages.append(chunk_message)  # save the message

      # print the time delay and text received
      full_reply_content = ''.join(collected_messages)
      return full_reply_content
    else:
      time_start = time.time()
      while self.num_retry < self.max_retry:
        with concurrent.futures.ThreadPoolExecutor() as executor:
          future = executor.submit(self.client.chat.completions.create,
                                   model=model,
                                   temperature=self.temp,
                                   messages=msg,
                                   logprobs=True,
                                   n=self.n)
          try:
            # Try to get the result of the function within the timeout
            result = future.result(timeout)
            time_end = time.time()
            print('time diff: ', time_end - time_start)
            return result
          except concurrent.futures.TimeoutError:
            # If the function exceeds the timeout, retry
            print(f"Function took longer than {timeout} seconds. Retrying...")
            time.sleep(1)
            self.num_retry += 1
      # completion = self.client.chat.completions.create(model=model,
      #                                                  temperature=self.temp,
      #                                                  messages=msg,
      #                                                  logprobs=True,
      #                                                  n=self.n)
      # return completion

  def build_input(self, hist: List[List[str]]):
    msg = []
    msg.append({"role": "system", "content": hist[0][0]})
    for h in hist[1:]:
      msg.append({"role": "user", "content": h[0]})
      if len(h) > 1:
        msg.append({"role": "assistant", "content": h[1]})

    return msg

  def __call__(self, checker: callable):
    js, hint = checker()
    for _ in range(self.MAX_RETRY_TIMES):
      reply = self._infer(self.build_input(hint))
      js, hint = checker(reply)
      if js is None:
        pass
        # print(hint[-1][-1])
      else:
        return js
    return None


class LLM_LLAMA_LOCAL(LLM):
  MAX_RETRY_TIMES = 5

  def __init__(self, nodes: list):
    self.present = 'LLaMA-Precise'
    self.nodes = nodes

    # concurrency control
    self.lock = threading.Lock()
    self.sema = threading.Semaphore(len(self.nodes))
    self.avail_list = [True for _ in range(len(self.nodes))]

  def _infer(self, func: str, data: dict):
    self.sema.acquire()
    with self.lock:
      idx = self.avail_list.index(True)
      self.avail_list[idx] = False

    if func == 'chat':
      api = self.nodes[idx]['chat']
    elif func == 'chateval':
      api = self.nodes[idx]['chateval']
    else:
      raise NotImplementedError
    response = rq.post(api, json=data)

    with self.lock:
      self.avail_list[idx] = True
    self.sema.release()
    return response

  def _chat(self, inp: str, history: List) -> str:
    data = {
        "user_input": inp,
        "history": {
            "internal": history,
            "visible": history
        },
        "mode": "chat",
        "preset": self.present,
        "instruction_template": "Llama-v2",
        "truncation_length": 4096,
    }
    # print(text)
    response = self._infer('chat', data)
    res = response.json()['results'][0]['history']['visible'][-1][-1]
    res = html.unescape(res).replace('"', '')
    response.json()['results'][0]['history']['visible'][-1][-1] = res
    return res

  def _continue(self, history: List[List[str]]):
    data = {
        "user_input": history[-1][0],
        "history": {
            "internal": history,
            "visible": history
        },
        "_continue": True,
        "mode": "chat",
        "preset": self.present,
        "instruction_template": "Llama-v2",
        "truncation_length": 4096,
    }
    # print(text)
    response = self._infer('chat', data)
    res = response.json()['results'][0]['history']['visible'][-1][-1]
    return res

  def eval_prob(self, prompt: List[List[str]], choice: List[str]):
    data = {
        "choices": choice,
        "history": {
            'internal': prompt,
            'visible': prompt
        },
        "mode": "chat",
        "preset": self.present,
        "instruction_template": "Llama-v2",
        "truncation_length": 4096,
    }
    response = self._infer('chateval', data)
    l = response.json()['ret']
    # print('l: ', l)
    # {'ret': [{'len': 10, 'logit': 4.5}, ...]}
    res = []
    for idx in range(1, len(l)):
      res.append(
          (l[idx]['logit'] - l[0]['logit']) / (l[idx]['len'] - l[0]['len']))
    return res

  def __call__(self, checker: callable):
    js, hint = checker()
    for _ in range(self.MAX_RETRY_TIMES):
      reply = self._chat(hint[-1][-1], hint[:-1])
      js, hint = checker(reply)
      if js is None:
        print("Error Output: ", reply)
        print("Error Hint: ", hint[-1][-1])
      else:
        return js
    return None
