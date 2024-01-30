import gradio as gr
from transformers import pipeline, AutoTokenizer
from peft import AutoPeftModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
loraModel = AutoPeftModelForSequenceClassification.from_pretrained("Intradiction/text_classification_WithLORA")
# Handle calls to DistilBERT no LORA
distilBERTnoLORA_pipe = pipeline(model="Intradiction/text_classification_NoLORA")
distilBERTwithLORA_pipe = pipeline("sentiment-analysis", model=loraModel, tokenizer=tokenizer)

def distilBERTnoLORA_fn(text):
    return distilBERTnoLORA_pipe(text)

def distilBERTwithLORA_fn(text):
    return distilBERTwithLORA_pipe(text)

def chat1(message,history):
    history = history or []
    message = message.lower()
    if message.startswith("how many"):
        response = ("1 to 10")
    else:
        response = ("whatever man whatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever manwhatever man")

    history.append((message, response))
    return history, history

chatbot = gr.Chatbot()
chatbot1 = gr.Chatbot()
chatbot2 = gr.Chatbot()

with gr.Blocks(
    title="",

) as demo:
    gr.Markdown("""
        <div style="overflow: hidden;color:#fff;display: flex;flex-direction: column;align-items: center; position: relative; width: 100%; height: 180px;background-size: cover; background-image: url(https://www.grssigns.co.uk/wp-content/uploads/web-Header-Background.jpg);">
            <img style="width: 130px;height: 60px;position: absolute;top:10px;left:10px" src="https://www.torontomu.ca/content/dam/tmumobile/images/TMU-Mobile-AppIcon.png"/>
            <span style="margin-top: 40px;font-size: 36px ;font-family:fantasy;">Efficient Fine tuning Of Large Language Models</span>
            <span style="margin-top: 10px;font-size: 14px;">By: Rahul Adams, Greylyn Gao, Rajevan Lograjh & Mahir Faisal</span>
            <span style="margin-top: 5px;font-size: 14px;">Group Id: AR06 FLC: Alice Reuada</span>
        </div>
    """)
    with gr.Tab("Text Classification"):
        with gr.Row():
            gr.Markdown("<h1>Efficient Fine Tuning for Text Classification</h1>")
        with gr.Row():
            with gr.Column(scale=0.3,variant="panel"):
                gr.Markdown("""
                            <h2>Specifciations</h2>
                            <p><b>Model:</b> Tiny Bert <br>
                            <b>Dataset:</b> IMDB Movie review dataset <br>
                            <b>NLP Task:</b> Text Classification</p>
                            <p>I don’t know why but I just enjoy doing this. Maybe it’s my way of dealing with stress or something but I just do it about once every week. Generally I’ll carry around a sack and creep around in a sort of crouch-walking position making goblin noises, then I’ll walk around my house and pick up various different “trinkets” and put them in my bag while saying stuff like “I’ll be having that” and laughing maniacally in my goblin voice (“trinkets” can include anything from stuff I find on the ground to cutlery or other utensils). The other day I was talking with my neighbours and they mentioned hearing weird noises like what I wrote about and I was just internally screaming the entire conversation.</p>
                            """)
                
            with gr.Column(scale=0.3,variant="panel"):
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

            with gr.Column():
                with gr.Row(variant="panel"):
                    out =  gr.Textbox(label= " DistilBERT no LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out1 = gr.Textbox(label= " DistilBERT with LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out2 = gr.Textbox(label= " LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

        btn.click(fn=distilBERTnoLORA_fn, inputs=inp, outputs=out)
        btn.click(fn=distilBERTwithLORA_fn, inputs=inp, outputs=out1)
        btn.click(fn=chat1, inputs=inp, outputs=out2)

    with gr.Tab("Natrual Language Infrencing"):
         with gr.Row():
             gr.Markdown("<h1>Efficient Fine Tuning for Natual Languae Infrencing</h1>")
         with gr.Row():
            with gr.Column(scale=0.3, variant="panel"):
                gr.Markdown("""
                            <h2>Specifciations</h2>
                            <p><b>Model:</b>  ELECTRA Bert Small <br>
                            <b>Dataset:</b> Stanford Natural Language Inference Dataset <br>
                            <b>NLP Task:</b> Natual Languae Infrencing</p>
                            <p>insert information on training parameters here</p>
                            """)
            with gr.Column(scale=0.3,variant="panel"):
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

            with gr.Column():
                with gr.Row(variant="panel"):
                    out =  gr.Textbox(label= " DistilBERT no LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out1 = gr.Textbox(label= " DistilBERT with LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out2 = gr.Textbox(label= " LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

    with gr.Tab("Sematic Text Similarity"):
         with gr.Row():
             gr.Markdown("<h1>Efficient Fine Tuning for Semantic Text Similarity</h1>")
         with gr.Row():
            with gr.Column(scale=0.3,variant="panel"):
                gr.Markdown("""
                            <h2>Specifciations</h2>
                            <p><b>Model:</b> DeBERTa-v3-xsmall <br>
                            <b>Dataset:</b> Quora Question Pairs dataset <br>
                            <b>NLP Task:</b> Semantic Text Similarity</p>
                            <p>insert information on training parameters here</p>
                            """)
            with gr.Column(scale=0.3,variant="panel"):
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

            with gr.Column():
                with gr.Row(variant="panel"):
                    out =  gr.Textbox(label= " DistilBERT no LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out1 = gr.Textbox(label= " DistilBERT with LoRA")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

                with gr.Row(variant="panel"):
                    out2 = gr.Textbox(label= " LoRA Fine Tuned Model")
                    gr.Markdown("""<div>
                                <span><center><B>Training Information</B><center></span>
                                <span><br><br><br><br><br></span>
                                </div>""")

    with gr.Tab("More information"):
        gr.Markdown("stuff to add")


if __name__ == "__main__":
    demo.launch()