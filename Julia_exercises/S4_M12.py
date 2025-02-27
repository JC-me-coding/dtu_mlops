#print(torch.__version__)
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity
from torch.profiler import profile, tensorboard_trace_handler


model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
   model(inputs)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

prof.export_chrome_trace("trace.json")

#print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

'''
with profile(...) as prof:
   for i in range(10):
      model(inputs)
      prof.step()
'''

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
   ...