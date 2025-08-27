import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import OpenAIEmbeddings

# Load .env and API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Constants
AUDIO_MODEL = "whisper-1"
LLAMA_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # example, replace with your model

print("Loading tokenizer and model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL).to(device)
print(f"Model loaded on {device}.")

# Initialize embeddings and DB globally for reuse
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
db = Chroma(persist_directory="db", embedding_function=embeddings)

def process_audio_file(audio_filepath):
    with open(audio_filepath, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model=AUDIO_MODEL,
            file=audio_file,
            response_format="text"
        )
    return transcript

def generate_meeting_minutes(transcript):
    prompt = f"Summarize the following meeting transcript:\n{transcript}\n\nMeeting Summary:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=500)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt repetition if any
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    return response

def chunk_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def add_summary_to_db(chunks, doc_id):
    documents = [Document(page_content=chunk, metadata={"source": doc_id}) for chunk in chunks]

    # Add to DB
    db.add_documents(documents)
    db.persist()

def transcribe_and_summarize(audio_filepath):
    transcript = process_audio_file(audio_filepath)
    summary = generate_meeting_minutes(transcript)
    chunks = chunk_text(summary)
    doc_id = os.path.basename(audio_filepath)
    add_summary_to_db(chunks, doc_id)
    return summary

def answer_question_from_db(question):
    # Search top 3 relevant chunks from DB
    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = (
        f"You are a helpful assistant. Use the following context from a meeting summary to answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if answer.startswith(prompt):
        answer = answer[len(prompt):].strip()

    return answer


# Gradio UI setup

with gr.Blocks() as demo:
    gr.Markdown("# Meeting Summarizer with Q&A")

    with gr.Tab("Upload & Summarize"):
        audio_input = gr.Audio(label="Upload Meeting Audio", type="filepath")
        summary_output = gr.Textbox(label="Meeting Minutes Summary", lines=20)
        summarize_btn = gr.Button("Transcribe & Summarize")

        summarize_btn.click(
            fn=transcribe_and_summarize,
            inputs=audio_input,
            outputs=summary_output
        )

    with gr.Tab("Q&A from Summary"):
        question_input = gr.Textbox(label="Ask a question about the meeting summary")
        answer_output = gr.Textbox(label="Answer", lines=10)
        ask_btn = gr.Button("Get Answer")

        ask_btn.click(
            fn=answer_question_from_db,
            inputs=question_input,
            outputs=answer_output
        )

if __name__ == "__main__":
    demo.launch()
