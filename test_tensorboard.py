
from tensorboard.backend.event_processing import event_accumulator
from tensorflow.python.summary.summary_iterator import summary_iterator
from huggingface_hub import hf_hub_download

tfevents_filepath = hf_hub_download(
    repo_id="Intradiction/BERT-SA-LORA", 
    filename="runs/Mar15_15-17-41_76d84c66d2d5/events.out.tfevents.1710515863.76d84c66d2d5.790.0"
)

ea = event_accumulator.EventAccumulator(tfevents_filepath,
size_guidance={ # see below regarding this argument
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})

ea.Reload()
print(ea.Tags())
#print(ea.Scalars('eval/loss'))
print(ea.FirstEventTimestamp())
print(ea.Scalars('eval/accuracy'))

print("===============================")

for e in summary_iterator(tfevents_filepath):
    print(e)