from transformers import pipeline
import gradio as gr

model = pipeline("summarization", model = 'facebook/bart-large-cnn')

def predict(prompt):
    summary = model(prompt)[0]['summary_text']
    return summary

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)

output = gr.Interface(fn=predict, inputs=textbox, outputs="text", title="News Summarizartion Demo with MLOps Techniques",)

output.launch()