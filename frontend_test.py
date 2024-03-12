import gradio as gr
from transformers import pipeline, AutoTokenizer
from peft.auto import AutoPeftModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
loraModel = AutoPeftModelForSequenceClassification.from_pretrained("Intradiction/text_classification_WithLORA")

tokenizer1 = AutoTokenizer.from_pretrained("albert-base-v2")



#pretrained models
#STSmodel_pipe = pipeline()
#NLImodel_pipe = pipeline()

# Handle calls to DistilBERT 
distilBERTUntrained_pipe = pipeline("sentiment-analysis", model="bert-base-uncased")
distilBERTnoLORA_pipe = pipeline(model="Intradiction/text_classification_NoLORA")
distilBERTwithLORA_pipe = pipeline("sentiment-analysis", model=loraModel, tokenizer=tokenizer)

#text class models 
def distilBERTnoLORA_fn(text):
    return distilBERTnoLORA_pipe(text)

def distilBERTwithLORA_fn(text):
    return distilBERTwithLORA_pipe(text)

def distilBERTUntrained_fn(text):
    return distilBERTUntrained_pipe(text)


# Handle calls to ALBERT
ALbertUntrained_pipe = pipeline("text-classification", model="albert-base-v2")
AlbertnoLORA_pipe = pipeline(model="Intradiction/NLI-Conventional-Fine-Tuning")
#AlbertwithLORA_pipe = pipeline()

#NLI models 
def AlbertnoLORA_fn(text1, text2):
    return AlbertnoLORA_pipe({'text': text1, 'text_pair': text2})

def AlbertwithLORA_fn(text1, text2):
    return ("working2")

def AlbertUntrained_fn(text1, text2):
    return ALbertUntrained_pipe({'text': text1, 'text_pair': text2})


# Handle calls to Deberta
DebertaUntrained_pipe = pipeline("text-classification", model="microsoft/deberta-v3-xsmall")
#DebertanoLORA_pipe = pipeline()
#DebertawithLORA_pipe = pipeline()

#STS models
def DebertanoLORA_fn(text1, text2):
    return ("working3")

def DebertawithLORA_fn(text1, text2):
    return ("working2")

def DebertaUntrained_fn(text1, text2):
    return DebertaUntrained_pipe({'text': text1, 'text_pair': text2})


#placeholder
def chat1(message,history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = ("1 to 10")
    else:
        response = ("whatever man whatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever man")

    history.append((message, response))
    return history, history


with gr.Blocks(
    title="",

) as demo:
    gr.Markdown("""
        <div style="overflow: hidden;color:#fff;display: flex;flex-direction: column;align-items: center; position: relative; width: 100%; height: 180px;background-size: cover; background-image: url(https://www.grssigns.co.uk/wp-content/uploads/web-Header-Background.jpg);">
            <img style="width: 130px;height: 60px;position: absolute;top:10px;left:10px" src="https://www.torontomu.ca/content/dam/tmumobile/images/TMU-Mobile-AppIcon.png"/>
            <span style="margin-top: 40px;font-size: 36px ;font-family:fantasy;">Efficient Fine Tuning Offf Large Language Models</span>
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
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    TextClassOut1 = gr.Textbox(label= "Conventionaly Trained Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    TextClassOut2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

        btn.click(fn=distilBERTUntrained_fn, inputs=inp, outputs=TextClassOut)
        btn.click(fn=distilBERTnoLORA_fn, inputs=inp, outputs=TextClassOut1)
        btn.click(fn=distilBERTwithLORA_fn, inputs=inp, outputs=TextClassOut2)
        

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
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    NLIOut1 = gr.Textbox(label= "Conventionaly Trained Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    NLIOut2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")
        
         nli_btn.click(fn=AlbertUntrained_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut)
         nli_btn.click(fn=AlbertnoLORA_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut1)
         nli_btn.click(fn=AlbertwithLORA_fn, inputs=[nli_p1,nli_p2], outputs=NLIOut2)

    with gr.Tab("Semantic Text Similarity"):
         with gr.Row():
             gr.Markdown("<h1>Efficient Fine Tuning for Semantic Text Similarity</h1>")
         with gr.Row():
            with gr.Column(variant="panel"):
                gr.Markdown("""
                            <h2>Specifications</h2>
                            <p><b>Model:</b> DeBERTa-v3-xsmall <br>
                            <b>Dataset:</b> Semantic Text Similarity Benchmark <br>
                            <b>NLP Task:</b> Semantic Text Similarity</p>
                            <p>Semantic text similarity measures the closeness in meaning of two pieces of text despite differences in their wording or structure. This task involves two input prompts which can be sentences, phrases or entire documents and assessing them for similarity. In our implementation we compare phrases represented by a score that can range between zero and one. A score of zero implies completely different phrases, while one indicates identical meaning between the text pair. This implementation uses a DeBERTa-v3-xsmall and training was performed on the semantic text similarity benchmark dataset which contains over 86k semantic pairs and their scores. We can see that when training is performed over [XX] epochs we see an increase in X% of training time for the LoRA trained model compared to a conventionally tuned model.</p>
                            """)
            with gr.Column(variant="panel"):
                sts_p1 = gr.Textbox(placeholder="Prompt One",label= "Enter Query")
                sts_p2 = gr.Textbox(placeholder="Prompt Two",label= "Enter Query")
                sts_btn = gr.Button("Run")
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
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    sts_out1 = gr.Textbox(label= "Conventionally Trained Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    sts_out2 = gr.Textbox(label= "LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Informadtion</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")
                    
         sts_btn.click(fn=DebertaUntrained_fn, inputs=[sts_p1,sts_p2], outputs=sts_out)
         sts_btn.click(fn=DebertanoLORA_fn, inputs=[sts_p1,sts_p2], outputs=sts_out1)
         sts_btn.click(fn=DebertawithLORA_fn, inputs=[sts_p1,sts_p2], outputs=sts_out2)

    with gr.Tab("More informatioen"):
        gr.Markdown("stuff to add")


if __name__ == "__main__":
    demo.launch()