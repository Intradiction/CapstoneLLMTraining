
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('events.out.tfevents.1701212945.784ae33ab242.985.0',
size_guidance={ # see below regarding this argument
event_accumulator.COMPRESSED_HISTOGRAMS: 500,
event_accumulator.IMAGES: 4,
event_accumulator.AUDIO: 4,
event_accumulator.SCALARS: 0,
event_accumulator.HISTOGRAMS: 1,
})

ea.Reload()
print(ea.Tags())
print(ea.Scalars('eval/loss'))