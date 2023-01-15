import torch
from time import time

DEFAULT_DATA = [torch.randn(16,16,16)]
DEFAULT_WORKLOAD = lambda data: data @ data
DEFAULT_DURATION = 0.2
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
DEFAULT_REPSIZE = 1000

def test(workload = DEFAULT_WORKLOAD, data = DEFAULT_DATA, dtypes = DEFAULT_DTYPES, duration = DEFAULT_DURATION, repsize = DEFAULT_REPSIZE, device = DEFAULT_DEVICE):
    results = []
    for dtype in dtypes:
        dtype_data = [item.to(dtype) for item in data]
        try:
            ct = 0
            start = time()
            now = start
            finish = start + duration
            while now < finish:
                for rep in range(repsize):
                    workload(*dtype_data)
                now = time()
                ct += 1
            rate = ct * repsize / (now - start) 
            results.append((rate, dtype))
        except:
            pass
    results.sort(reverse=True)
    return results

if __name__ == '__main__':
    for rate, dtype in test(device='cpu'):
        print(f'cpu {dtype}: {rate} works/sec')
