import json
import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModel, BertForSequenceClassification, AlbertForSequenceClassification, DebertaForSequenceClassification, AutoModelForSequenceClassification, RobertaForSequenceClassification
from peft.auto import AutoPeftModelForSequenceClassification
from tensorboard.backend.event_processing import event_accumulator
from peft import PeftModel
from huggingface_hub import hf_hub_download
import plotly.express as px
import pandas as pd

# Parse sentiment analysis pipeline results
def parse_pipe_sa(pipe_out_text: str):
    output_list = list(pipe_out_text)
    pipe_label = output_list[0]['label']
    pipe_score = output_list[0]['score']

    parsed_prediction = 'NULL'

    if pipe_label == 'NEGATIVE' or pipe_label == 'LABEL_0':
        parsed_prediction = f'This model thinks the sentiment is negative with a confidence score of {pipe_score}'
    elif pipe_label == 'POSITIVE' or pipe_label == 'LABEL_1':
        parsed_prediction = f'This model thinks the sentiment is positive with a confidence score of {pipe_score}'

    return parsed_prediction

loraModel = AutoPeftModelForSequenceClassification.from_pretrained("Intradiction/text_classification_WithLORA")
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer1 = AutoTokenizer.from_pretrained("albert-base-v2")
tokenizer2 = AutoTokenizer.from_pretrained("microsoft/deberta-v3-xsmall")


# Handle calls to DistilBERT------------------------------------------
base_model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
peft_model_id = "Intradiction/BERT-SA-LORA"
model = PeftModel.from_pretrained(model=base_model, model_id=peft_model_id)
sa_merged_model = model.merge_and_unload()
bbu_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

distilBERTUntrained_pipe = pipeline("sentiment-analysis", model="bert-base-uncased")
distilBERTnoLORA_pipe = pipeline(model="Intradiction/text_classification_NoLORA")
SentimentAnalysis_LORA_pipe = pipeline("sentiment-analysis", model=sa_merged_model, tokenizer=bbu_tokenizer)

#text class models 
def distilBERTnoLORA_fn(text):
    return parse_pipe_sa(distilBERTnoLORA_pipe(text))

def distilBERTwithLORA_fn(text):
    return parse_pipe_sa(SentimentAnalysis_LORA_pipe(text))

def distilBERTUntrained_fn(text):
    return parse_pipe_sa(distilBERTUntrained_pipe(text))


# Handle calls to ALBERT---------------------------------------------
base_model1 = AlbertForSequenceClassification.from_pretrained("Alireza1044/albert-base-v2-mnli")
peft_model_id1 = "m4faisal/NLI-Lora-Fine-Tuning-10K-ALBERT"
model1 = PeftModel.from_pretrained(model=base_model1, model_id=peft_model_id1)
sa_merged_model1 = model1.merge_and_unload()
bbu_tokenizer1 = AutoTokenizer.from_pretrained("Alireza1044/albert-base-v2-mnli")

ALbertUntrained_pipe = pipeline("text-classification", model="Alireza1044/albert-base-v2-mnli")
AlbertnoLORA_pipe = pipeline(model="m4faisal/NLI-Conventional-Fine-Tuning")
AlbertwithLORA_pipe = pipeline("text-classification",model=sa_merged_model1, tokenizer=bbu_tokenizer1)

#NLI models 
def AlbertnoLORA_fn(text1, text2):
    return AlbertnoLORA_pipe({'text': text1, 'text_pair': text2})

def AlbertwithLORA_fn(text1, text2):
    return AlbertwithLORA_pipe({'text': text1, 'text_pair': text2})

def AlbertUntrained_fn(text1, text2):
    return ALbertUntrained_pipe({'text': text1, 'text_pair': text2})


# Handle calls to Deberta--------------------------------------------
base_model2 = RobertaForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=3)
peft_model_id2 = "rajevan123/STS-Lora-Fine-Tuning-Capstone-roberta-base-filtered-137-with-higher-r-mid"
model2 = PeftModel.from_pretrained(model=base_model2, model_id=peft_model_id2)
sa_merged_model2 = model2.merge_and_unload()
bbu_tokenizer2 = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

DebertaUntrained_pipe = pipeline("text-classification", model="FacebookAI/roberta-base")
DebertanoLORA_pipe = pipeline(model="rajevan123/STS-conventional-Fine-Tuning-Capstone-roberta-base-filtered-137")
DebertawithLORA_pipe = pipeline("text-classification",model=sa_merged_model2, tokenizer=bbu_tokenizer2)

#STS models
def DebertanoLORA_fn(text1, text2):
    return DebertanoLORA_pipe({'text': text1, 'text_pair': text2})

def DebertawithLORA_fn(text1, text2):
    return DebertawithLORA_pipe({'text': text1, 'text_pair': text2})
    #return ("working2")

def DebertaUntrained_fn(text1, text2):
    return DebertaUntrained_pipe({'text': text1, 'text_pair': text2})

#helper functions ------------------------------------------------------

#Text metrics for Untrained models
def displayMetricStatsUntrained():
    return "No statistics to display for untrained models"

def displayMetricStatsText():
    #file_name = 'events.out.tfevents.distilbertSA-conventional.0'
    file_name = hf_hub_download(repo_id="Intradiction/text_classification_NoLORA", filename="runs/Nov28_21-52-51_81dc5cd53c46/events.out.tfevents.1701208378.81dc5cd53c46.1934.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')

    #code to pull time data (very inaccurate)
    # time_data = event_acc.Scalars('eval/runtime')
    # Ttime = 0 
    # for time in time_data:
    #     Ttime+=time.value
    # Ttime = str(round(Ttime/60,2))
    # print(Ttime)

    metrics = ("Active Training Time: 27.95 mins \n\n")
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics

def displayMetricStatsTextTCLora():
    #file_name = 'events.out.tfevents.distilbertSA-LORA.0'
    file_name = hf_hub_download(repo_id="Intradiction/BERT-SA-LORA", filename="runs/Mar16_18-10-29_INTRADICTION/events.out.tfevents.1710627034.INTRADICTION.31644.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')
    
    #code to pull time data (very inaccurate)
    # time_data = event_acc.Scalars('eval/runtime')
    # Ttime = 0 
    # for time in time_data:
    #     Ttime+=time.value
    # Ttime = str(round(Ttime/60,2))
    # print(event_acc.Tags())

    metrics = ("Active Training Time: 15.58 mins \n\n")
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics

def displayMetricStatsTextNLINoLora():
    #file_name = 'events.out.tfevents.NLI-Conventional.1'
    file_name = hf_hub_download(repo_id="m4faisal/NLI-Conventional-Fine-Tuning", filename="runs/Mar20_23-18-22_a7cbf6b28344/events.out.tfevents.1710976706.a7cbf6b28344.5071.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')
    metrics = "Active Training Time: 6.74 mins \n\n"
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics

def displayMetricStatsTextNLILora():
    #file_name = 'events.out.tfevents.NLI-Lora.0'
    file_name = hf_hub_download(repo_id="m4faisal/NLI-Lora-Fine-Tuning-10K", filename="runs/Mar20_18-07-52_87caf1b1d04f/events.out.tfevents.1710958080.87caf1b1d04f.7531.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')
    metrics = "Active Training Time: 15.04 mins \n\n"
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics

def displayMetricStatsTextSTSLora():
    #file_name = 'events.out.tfevents.STS-Lora.2'
    file_name = hf_hub_download(repo_id="rajevan123/STS-Lora-Fine-Tuning-Capstone-roberta-base-filtered-137-with-higher-r-mid", filename="runs/Mar28_19-51-13_fcdc58e67935/events.out.tfevents.1711655476.fcdc58e67935.625.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')
    metrics = "Active Training Time: 41.07 mins \n\n"
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics
def displayMetricStatsTextSTSNoLora():
    #file_name = 'events.out.tfevents.STS-Conventional.0'
    file_name = hf_hub_download(repo_id="rajevan123/STS-conventional-Fine-Tuning-Capstone-roberta-base-filtered-137", filename="runs/Mar31_15-13-28_585e70ba99a4/events.out.tfevents.1711898010.585e70ba99a4.247.0")
    event_acc = event_accumulator.EventAccumulator(file_name,
    size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
    event_acc.Reload()
    accuracy_data = event_acc.Scalars('eval/accuracy')
    loss_data = event_acc.Scalars('eval/loss')
    metrics = "Active Training Time: 23.96 mins \n\n"
    for i in range(0, len(loss_data)):
        metrics = metrics + 'Epoch Number: ' + str(i) + '\n'
        metrics = metrics + 'Accuracy (%): ' + str(round(accuracy_data[i].value * 100, 3)) + '\n'
        metrics = metrics + 'Loss (%): ' + str(round(loss_data[i].value * 100, 3)) + '\n\n'
    
    return metrics

def displayMetricStatsGraph():
   file_name = 'events.out.tfevents.1701212945.784ae33ab242.985.0'
   event_acc = event_accumulator.EventAccumulator(file_name,
   size_guidance={
    event_accumulator.COMPRESSED_HISTOGRAMS: 500,
    event_accumulator.IMAGES: 4,
    event_accumulator.AUDIO: 4,
    event_accumulator.SCALARS: 0,
    event_accumulator.HISTOGRAMS: 1,
})
   
   event_acc.Reload()
   accuracy_data = event_acc.Scalars('eval/accuracy')
   loss_data = event_acc.Scalars("eval/loss")
   epoch = []
   metric = []
   group = []
   for i in range(0, len(accuracy_data)):
       epoch.append(str(i))
       metric.append(accuracy_data[i].value)
       group.append('G1')
   for j in range(0, len(loss_data)):
       epoch.append(str(j))
       metric.append(loss_data[j].value)
       group.append('G2')
   data = pd.DataFrame()
   data['Epoch'] = epoch
   data['Metric'] = metric
   data['Group'] = group

  #generate the actual plot
   return px.line(data, x = 'Epoch', y = 'Metric', color=group, markers = True)


# #placeholder
# def chat1(message,history):
#     history = history or []
#     message = message.lower()
#     if message.startswith("how many"):
#         response = ("1 to 10")
#     else:
#         response = ("whatever man whatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever man")
#     history.append((message, response))
#     return history, history


with gr.Blocks(
    title="",

) as demo:
    gr.Markdown("""
        <div style="overflow: hidden;color:#fff;display: flex;flex-direction: column;align-items: center; position: relative; width: 100%; height: 180px;background-size: cover; background-image: url(https://www.grssigns.co.uk/wp-content/uploads/web-Header-Background.jpg);">
            <img style="width: 130px;height: 60px;position: absolute;top:10px;left:10px" src="https://www.torontomu.ca/content/dam/tmumobile/images/TMU-Mobile-AppIcon.png"/>
            <span style="margin-top: 40px;font-size: 36px ;font-family:fantasy;">Efficient Fine Tuning Of Large Language Models</span>
            <span style="margin-top: 10px;font-size: 14px;">By: Rahul Adams, Greylyn Gao, Rajevan Logarajah & Mahir Faisal</span>
            <span style="margin-top: 5px;font-size: 14px;">Group Id: AR06 FLC: Alice Reuda</span>
        </div>
    """)
    with gr.Tab("Text Classification"):
        with gr.Row():
            gr.Markdown("<h1>Efficient Fine Tuning for Text Classification</h1>")
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("""
                            <h2>Specifications</h2>
                            <p><b>Model:</b> Tiny Bert <br>
                            <b>Dataset:</b> IMDB Movie review dataset <br>
                            <b>NLP Task:</b> Text Classification</p>
                            <p>Text classification is an NLP task that focuses on automatically ascribing a predefined category or labels to an input prompt. In this demonstration the Tiny Bert model has been used to classify the text on the basis of sentiment analysis, where the labels (negative and positive) will indicate the emotional state expressed by the input prompt. The tiny bert model was chosen as in its base state its ability to perform sentiment analysis is quite poor, displayed by the untrained model, which often fails to correctly ascribe the label to the sentiment. The models were trained on the IMDB dataset which includes over 100k sentiment pairs pulled from IMDB movie reviews. We can see that when training is performed over [XX] of epochs we see an increase in X% of training time for the LoRA trained model.</p>
                            """)
                
            with gr.Column(variant="panel"):
                inp = gr.Textbox(placeholder="Prompt",label= "Enter Query")
                btn = gr.Button("Run")
                btnTextClassStats = gr.Button("Display Training Metrics")
                btnTensorLinkTCNoLora = gr.Button(value="View Conventional Training Graphs", link="https://huggingface.co/Intradiction/text_classification_NoLORA/tensorboard")
                btnTensorLinkTCLora = gr.Button(value="View LoRA Training Graphs", link="https://huggingface.co/Intradiction/BERT-SA-LORA/tensorboard")
                gr.Examples(
                    [
                        "I thought this was a bit contrived",
                        "You would need to be a child to enjoy this",
                        "Drive more like Drive away",
                    ],
                    inp,
                    label="Try asking",
                )

            with gr.Column(scale=3):
                with gr.Row(variant="panel"):
                    TextClassOut =  gr.Textbox(label= "Untrained Base Model")
                    TextClassUntrained = gr.Textbox(label = "Training Informaiton")

                with gr.Row(variant="panel"):
                    TextClassOut1 = gr.Textbox(label="Conventionaly Trained Model")
                    TextClassNoLoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 27.95 mins")

                with gr.Row(variant="panel"):
                    TextClassOut2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    TextClassLoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 15.58 mins")

        btn.click(fn=distilBERTUntrained_fn, inputs=inp, outputs=TextClassOut)
        btn.click(fn=distilBERTnoLORA_fn, inputs=inp, outputs=TextClassOut1)
        btn.click(fn=distilBERTwithLORA_fn, inputs=inp, outputs=TextClassOut2)
        btnTextClassStats.click(fn=displayMetricStatsUntrained, outputs=TextClassUntrained)
        btnTextClassStats.click(fn=displayMetricStatsText, outputs=TextClassNoLoraStats)
        btnTextClassStats.click(fn=displayMetricStatsTextTCLora, outputs=TextClassLoraStats) 

    with gr.Tab("Natural Language Inferencing"):
         with gr.Row():
             gr.Markdown("<h1>Efficient Fine Tuning for Natural Language Inferencing</h1>")
         with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("""
                            <h2>Specifications</h2>
                            <p><b>Model:</b> Albert <br>
                            <b>Dataset:</b> Stanford Natural Language Inference Dataset <br>
                            <b>NLP Task:</b> Natual Languae Infrencing</p>
                            <p>Natural Language Inference (NLI) which can also be referred to as Textual Entailment is an NLP task with the objective of determining the relationship between two pieces of text. In this demonstration the Albert model has been used to determine textual similarity ascribing a correlation score by the comparison of the two input prompts to determine if. Albert was chosen due to its substandard level of performance in its base state allowing room for improvement during training. The models were trained on the Stanford Natural Language Inference Dataset is a collection of 570k human-written English sentence pairs manually labeled for balanced classification, listed as positive, negative or neutral. We can see that when training is performed over [XX] epochs we see an increase in X% of training time for the LoRA trained model compared to a conventionally tuned model. </p>
                            """)
            with gr.Column(variant="panel"):
                nli_p1 = gr.Textbox(placeholder="Prompt One",label= "Enter Query")
                nli_p2 = gr.Textbox(placeholder="Prompt Two",label= "Enter Query")
                nli_btn = gr.Button("Run")
                btnNLIStats = gr.Button("Display Training Metrics")
                btnTensorLinkNLICon = gr.Button(value="View Conventional Training Graphs", link="https://huggingface.co/m4faisal/NLI-Conventional-Fine-Tuning/tensorboard") 
                btnTensorLinkNLILora = gr.Button(value="View LoRA Training Graphs", link="https://huggingface.co/m4faisal/NLI-Lora-Fine-Tuning-10K/tensorboard")
                gr.Examples(
                    [
                        "I am with my friends",
                        "People like apples",
                        "Dogs like bones",
                    ],
                    nli_p1,
                    label="Try asking",
                ) 
                gr.Examples(
                    [
                        "I am happy",
                        "Apples are good",
                        "Bones like dogs",
                    ],
                    nli_p2,
                    label="Try asking",
                ) 

            with gr.Column(scale=3):
                with gr.Row(variant="panel"):
                    NLIOut =  gr.Textbox(label= "Untrained Base Model")
                    NLIUntrained = gr.Textbox(label = "Training Informaiton")

                with gr.Row(variant="panel"):
                    NLIOut1 = gr.Textbox(label= "Conventionaly Trained Model")
                    NLINoLoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 6.74 mins")

                with gr.Row(variant="panel"):
                    NLIOut2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    NLILoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 15.04 mins")
        
         nli_btn.click(fn=AlbertUntrained_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut)
         nli_btn.click(fn=AlbertnoLORA_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut1)
         nli_btn.click(fn=AlbertwithLORA_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut2)
         btnNLIStats.click(fn=displayMetricStatsUntrained, outputs=NLIUntrained)
         btnNLIStats.click(fn=displayMetricStatsTextNLINoLora, outputs=NLINoLoraStats)
         btnNLIStats.click(fn=displayMetricStatsTextNLILora, outputs=NLILoraStats)
         

    with gr.Tab("Semantic Text Similarity"):
         with gr.Row():
             gr.Markdown("<h1>Efficient Fine Tuning for Semantic Text Similarity</h1>")
         with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("""
                            <h2>Specifications</h2>
                            <p><b>Model:</b> Roberta Base <br>
                            <b>Dataset:</b> Semantic Text Similarity Benchmark <br>
                            <b>NLP Task:</b> Semantic Text Similarity</p>
                            <p>Semantic text similarity measures the closeness in meaning of two pieces of text despite differences in their wording or structure. This task involves two input prompts which can be sentences, phrases or entire documents and assessing them for similarity. In our implementation we compare phrases represented by a score that can range between zero and one. A score of zero implies completely different phrases, while one indicates identical meaning between the text pair. This implementation uses a DeBERTa-v3-xsmall and training was performed on the semantic text similarity benchmark dataset which contains over 86k semantic pairs and their scores. We can see that when training is performed over [XX] epochs we see an increase in X% of training time for the LoRA trained model compared to a conventionally tuned model.</p>
                            """)
            with gr.Column(variant="panel"):
                sts_p1 = gr.Textbox(placeholder="Prompt One",label= "Enter Query")
                sts_p2 = gr.Textbox(placeholder="Prompt Two",label= "Enter Query")
                sts_btn = gr.Button("Run")
                btnSTSStats = gr.Button("Display Training Metrics")
                btnTensorLinkSTSCon = gr.Button(value="View Conventional Training Graphs", link="https://huggingface.co/rajevan123/STS-Conventional-Fine-Tuning/tensorboard")
                btnTensorLinkSTSLora = gr.Button(value="View Lora Training Graphs", link="https://huggingface.co/rajevan123/STS-Lora-Fine-Tuning-Capstone-roberta-base-filtered-137-with-higher-r-mid/tensorboard")
                gr.Examples(
                    [
                        "the ball is green",
                        "i dont like apples",
                        "our air is clean becase of trees",
                    ],
                    sts_p1,
                    label="Try asking",
                )
                gr.Examples(
                    [
                        "the green ball",
                        "apples are great",
                        "trees produce oxygen",
                    ],
                    sts_p2,
                    label="Try asking",
                )

            with gr.Column(scale=3):
                with gr.Row(variant="panel"):
                    sts_out =  gr.Textbox(label= "Untrained Base Model")
                    STSUntrained = gr.Textbox(label = "Training Informaiton")

                with gr.Row(variant="panel"):
                    sts_out1 = gr.Textbox(label= "Conventionally Trained Model")
                    STSNoLoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 23.96 mins")

                with gr.Row(variant="panel"):
                    sts_out2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    STSLoraStats = gr.Textbox(label = "Training Informaiton - Active Training Time: 14.62 mins")
                    
         sts_btn.click(fn=DebertaUntrained_fn, inputs=[sts_p1,sts_p2], outputs=sts_out)
         sts_btn.click(fn=DebertanoLORA_fn, inputs=[sts_p1,sts_p2], outputs=sts_out1)
         sts_btn.click(fn=DebertawithLORA_fn, inputs=[sts_p1,sts_p2], outputs=sts_out2)
         btnSTSStats.click(fn=displayMetricStatsUntrained, outputs=STSUntrained)
         btnSTSStats.click(fn=displayMetricStatsTextSTSNoLora, outputs=STSNoLoraStats)
         btnSTSStats.click(fn=displayMetricStatsTextSTSLora, outputs=STSLoraStats)

    with gr.Tab("More information"):
        gr.Markdown("stuff to add")


if __name__ == "__main__":
    demo.launch()