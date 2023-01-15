import torch
import warnings
from time import time

DEFAULT_DATA = [torch.randn(32,32,32),torch.randn(32,32,32)]
DEFAULT_WORKLOAD = lambda data1, data2: data1 @ data2
DEFAULT_DURATION = 10.0
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
DEFAULT_REPSIZE = 1024
DEFAULT_WARMUPS = 1

def test(workload = DEFAULT_WORKLOAD, data = DEFAULT_DATA, dtypes = DEFAULT_DTYPES, duration = DEFAULT_DURATION, repsize = DEFAULT_REPSIZE, warmups = DEFAULT_WARMUPS, device = DEFAULT_DEVICE):
    results = []
    duration /= len(dtypes)
    for dtype in dtypes:
        dtype_data = [item.to(dtype) for item in data]
        try:
            dtype_data = [item.to(device) for item in data]
            ct = 0
            start = time()
            finish = start + duration
            now = start
            for warmup in range(warmups):
                for rep in range(repsize):
                    workload(*dtype_data)
            now = time()
            if now >= finish:
                warnings.warn(f'{dtype} warmup took {now - start}/{duration} sec', stacklevel=2)
                finish = now + duration
            start = now
            while now < finish:
                for rep in range(repsize):
                    workload(*dtype_data)
                now = time()
                ct += 1
            rate = ct * repsize / (now - start) 
            results.append((dtype, rate))
        except Exception as exception:
            results.append((dtype, exception))
    results.sort(reverse=True, key = lambda tuple: tuple[1] if type(tuple[1]) is float else 0)
    return results

if __name__ == '__main__':
    for dtype, rate in test(device='cpu'):
        print(f'cpu {dtype}: {rate} works/sec')
