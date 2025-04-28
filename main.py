import nest_asyncio
import chromadb
import hashlib
import os
import json
import io
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.prompts import PromptTemplate
import fitz  # PyMuPDF
import cv2
import numpy as np
from gtts import gTTS
import re
from PIL import Image
import pytesseract
from fuzzywuzzy import process
import uuid
from datetime import datetime
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from pdf2image import convert_from_path
import pandas as pd
from azure.storage.blob import BlobServiceClient
import uvicorn


# Load environment variables
load_dotenv()

# Initialize Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))

def download_blob(container, blob_name):
    """Download a blob from Azure Blob Storage."""
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    try:
        blob_data = blob_client.download_blob().readall()
        return blob_data
    except Exception as e:
        print(f"Error downloading blob {blob_name} from container {container}: {e}")
        raise

def upload_to_blob(container, file_path, blob_name):
    """Upload a file to Azure Blob Storage and return the URL."""
    blob_client = blob_service_client.get_blob_client(container=container, blob=blob_name)
    try:
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        return blob_client.url
    except Exception as e:
        print(f"Error uploading blob {blob_name} to container {container}: {e}")
        raise

# Initialize ChromaDB client (use Blob Storage path or local fallback)
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH", "./chroma_db"))

# In-memory cache for query results
QUERY_CACHE = {}

# Metadata and JSON files (load from Blob Storage)
METADATA_FILE = "pdf_metadata.json"
JSON_FILE = "query_responses_4.json"

def multiple_image(query):
    """Extract images from PDFs based on query and upload to Blob Storage."""
    page_list = []
    output_folder = "/tmp/output_images"  # Temporary local directory
    os.makedirs(output_folder, exist_ok=True)

    # Load Excel data from Blob Storage
    excel_data = download_blob("data", "cliplevel.xlsx")
    df = pd.read_excel(io.BytesIO(excel_data))

    # Loop through rows
    query = query.lower()
    for _, row in df.iterrows():
        question = str(row['Question']).strip()
        if question == query:
            pdf_path = str(row['PDF_path']).strip()
            page_numbers = str(row['page_no']).strip()

            if not question or not pdf_path or not page_numbers:
                continue

            # Download PDF from Blob Storage
            pdf_data = download_blob("uploads", pdf_path)
            temp_pdf_path = f"/tmp/{os.path.basename(pdf_path)}"
            with open(temp_pdf_path, "wb") as f:
                f.write(pdf_data)

            try:
                page_list = [int(p.strip()) for p in page_numbers.split(',')]
            except ValueError:
                print(f"❌ Invalid page numbers: {page_numbers}")
                continue

            for page_num in page_list:
                try:
                    images = convert_from_path(
                        temp_pdf_path,
                        first_page=page_num,
                        last_page=page_num
                    )
                    for img in images:
                        safe_question = "".join(c if c.isalnum() else "_" for c in question)[:50]
                        img_name = f"{safe_question}_page_{page_num}.png"
                        img_path = os.path.join(output_folder, img_name)
                        img.save(img_path)
                        # Upload to Blob Storage
                        blob_url = upload_to_blob("output_images", img_path, img_name)
                        print(f"✅ Saved to Blob Storage: {blob_url}")
                except Exception as e:
                    print(f"❌ Error processing {pdf_path} page {page_num}: {e}")
    return page_list

# Function to initialize or load JSON file
def init_json_file():
    """Initialize query_responses_4.json if it doesn't exist in Blob Storage."""
    try:
        download_blob("data", JSON_FILE)
    except:
        # Create an empty JSON file and upload
        empty_data = json.dumps({})
        blob_client = blob_service_client.get_blob_client(container="data", blob=JSON_FILE)
        blob_client.upload_blob(empty_data, overwrite=True)

# Function to initialize or load metadata file
def init_metadata_file():
    """Initialize pdf_metadata.json if it doesn't exist in Blob Storage."""
    try:
        download_blob("data", METADATA_FILE)
    except:
        # Create an empty JSON file and upload
        empty_data = json.dumps({})
        blob_client = blob_service_client.get_blob_client(container="data", blob=METADATA_FILE)
        blob_client.upload_blob(empty_data, overwrite=True)

# Function to load PDF metadata
def load_pdf_metadata():
    """Load pdf_metadata.json from Blob Storage."""
    init_metadata_file()
    metadata_data = download_blob("data", METADATA_FILE)
    return json.loads(metadata_data)

# Function to load vector index for a single PDF
def load_vector_index(file_path, embed_model):
    try:
        metadata = load_pdf_metadata()
        if file_path not in metadata:
            print(f"No index found for {file_path}. Run pdf_indexer.py first.")
            return None
        collection_name = metadata[file_path]["collection_name"]
        chroma_collection = chroma_client.get_collection(collection_name)
        print(f"Collection {collection_name} has {chroma_collection.count()} documents.")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        vector_index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
        print(f"Loaded vector index for {file_path} from collection {collection_name}.")
        return vector_index
    except Exception as e:
        print(f"Error loading vector index for {file_path}: {e}")
        return None

# Function to format response and extract metadata
def format_response(response_text, file_path, query_text):
    page_number = response_text['metadata'].get("page_number", "Unknown")
    query_lower = query_text.lower()
    if any(keyword in query_lower for keyword in ["who", "name"]):
        match = re.search(r'([A-Za-z\s]+)', response_text['text'])
        name = match.group(0).strip() if match else "Not specified"
        return name, page_number
    elif any(keyword in query_lower for keyword in ["process", "steps", "flow"]):
        steps = re.findall(r'(.+?)(?=\n\d+\.|$|\n[A-Za-z])', response_text['text'], re.DOTALL)
        if steps:
            return "\n".join(step.strip() for step in steps if step.strip()), page_number
        return "No process steps found", page_number
    else:
        return response_text['text'].strip(), page_number

# Function to handle the query for a single PDF
def query_document(vector_index, query_text, llm, file_path):
    try:
        qa_template = PromptTemplate(
            "Given the context, provide a precise answer to the question. Ensure the answer is directly relevant to the query and extracted from the provided context. If the question asks for a specific process (e.g., 'Invoice from Ariba Network'), return only the relevant steps or details from the context. If no relevant information is found, return 'No relevant information found.' Context: {context_str}\nQuestion: {query_str}\nAnswer:"
        )
        query_engine = vector_index.as_query_engine(
            llm=llm,
            response_mode="compact",
            similarity_top_k=10,
            text_qa_template=qa_template
        )
        response = query_engine.query(query_text)
        if not response.source_nodes:
            print(f"No matching document found in {file_path}.")
            return None, None, None
        metadata = response.source_nodes[0].metadata
        page_number = metadata.get("page_number")
        score = response.source_nodes[0].score if hasattr(response.source_nodes[0], 'score') else 0
        if not page_number:
            print(f"No page_number found in metadata for {file_path}.")
            return None, None, None
        print(f"Retrieved context for {file_path} (page {page_number}, score {score}):\n{response.source_nodes[0].text[:500]}")
        formatted_response, _ = format_response({"text": str(response), "metadata": metadata}, file_path, query_text)
        return formatted_response, page_number, score
    except Exception as e:
        print(f"Error querying document {file_path}: {e}")
        return None, None, None

# Function to extract text from scanned PDFs (OCR)
def extract_text_from_image(pdf_path, page_number):
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert("L")  # Convert to grayscale
        img = img.point(lambda x: 0 if x < 128 else 255, "1")  # Binarize
        text = pytesseract.image_to_string(img)
        print(f"OCR Extracted Text for {pdf_path} (page {page_number}):\n{text[:500]}")
        return text
    except Exception as e:
        print(f"Error performing OCR on {pdf_path} (page {page_number}): {e}")
        return ""

# Function to find similar text for highlighting
def find_similar_text(page, query):
    text = page.get_text("text")
    lines = text.split("\n")
    best_match = process.extractOne(query, lines, score_cutoff=80)  # Higher threshold
    return best_match[0] if best_match else None

# Function to extract and highlight the answer in an image
def extract_and_highlight(file_path, page_number, answer, query_id, IMAGE_DIR):
    try:
        doc = fitz.open(file_path)
        page = doc.load_page(page_number - 1)
        extracted_text = page.get_text("text")
        if not extracted_text.strip():
            print(f"Page {page_number} has no extracted text. Trying OCR...")
            extracted_text = extract_text_from_image(file_path, page_number)
            if not extracted_text.strip():
                print(f"OCR failed to extract text for {file_path} (page {page_number}).")
                return None
        print(f"Extracted Text on Page {page_number} for {file_path}:\n{extracted_text[:500]}")
        text_instances = page.search_for(answer)
        if not text_instances:
            print(f"No exact match found for: {answer}")
            approx_answer = find_similar_text(page, answer)
            if approx_answer:
                print(f"Using fuzzy match: {approx_answer}")
                text_instances = page.search_for(approx_answer)
            else:
                print(f"No fuzzy match found for {file_path} (page {page_number}).")
                return None
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        for rect in text_instances:
            x0, y0, x1, y1 = int(rect.x0), int(rect.y0), int(rect.x1), int(rect.y1)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        highlighted_img = Image.fromarray(img)
        img_path = os.path.join(IMAGE_DIR, f"{query_id}_highlighted.png")
        highlighted_img.save(img_path)
        # Upload to Blob Storage
        blob_url = upload_to_blob("output_images", img_path, f"{query_id}_highlighted.png")
        print(f"Highlighted answer uploaded to Blob Storage: {blob_url}")
        return blob_url
    except Exception as e:
        print(f"Error extracting and highlighting answer for {file_path}: {e}")
        return None

# Function to create audio response
def generate_audio(response_text, query_id, AUDIO_DIR):
    try:
        tts = gTTS(text=response_text, lang="en")
        audio_path = os.path.join(AUDIO_DIR, f"{query_id}_response.mp3")
        tts.save(audio_path)
        # Upload to Blob Storage
        blob_url = upload_to_blob("output_audio", audio_path, f"{query_id}_response.mp3")
        print(f"Audio response uploaded to Blob Storage: {blob_url}")
        return blob_url
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# Function to save query and response to JSON
def save_to_json(query_id, query_text, response, page_number=None, img_path=None, audio_path=None, pdf_path=None):
    init_json_file()
    json_data = download_blob("data", JSON_FILE)
    data = json.loads(json_data)
    data[query_id] = {
        "query": query_text,
        "response": response,
        "page_number": page_number,
        "image_path": os.path.basename(img_path) if img_path else None,
        "audio_path": os.path.basename(audio_path) if audio_path else None,
        "pdf_path": os.path.basename(pdf_path) if pdf_path else None,
        "timestamp": str(datetime.now())
    }
    # Upload updated JSON back to Blob Storage
    blob_client = blob_service_client.get_blob_client(container="data", blob=JSON_FILE)
    blob_client.upload_blob(json.dumps(data, indent=4), overwrite=True)
    return data[query_id]

# Function to check cached query (case-insensitive, with expiration)
def check_cached_query(query_text, max_age_hours=24):
    query_text_lower = query_text.lower()
    cache_key = query_text_lower
    if cache_key in QUERY_CACHE:
        print(f"Returning in-memory cached response for query: {query_text}")
        return QUERY_CACHE[cache_key]
    init_json_file()
    json_data = download_blob("data", JSON_FILE)
    data = json.loads(json_data)
    for query_id, entry in data.items():
        if entry["query"].lower() == query_text_lower:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            age = (datetime.now() - timestamp).total_seconds() / 3600
            if age < max_age_hours:
                print(f"Returning JSON-cached response for query: {query_text}")
                QUERY_CACHE[cache_key] = {
                    "response": entry["response"],
                    "page_number": entry["page_number"],
                    "image_path": entry["image_path"],
                    "audio_path": entry["audio_path"],
                    "pdf_path": entry.get("pdf_path")
                }
                return QUERY_CACHE[cache_key]
    return None

# Function to handle queries
def process_query(query_text, embed_model, llm, IMAGE_DIR, AUDIO_DIR):
    cached_response = check_cached_query(query_text)
    if cached_response:
        return {
            "query": query_text,
            "response": cached_response["response"],
            "page_number": cached_response["page_number"],
            "image_path": cached_response["image_path"],
            "audio_path": cached_response["audio_path"],
            "pdf_path": cached_response["pdf_path"],
            "timestamp": datetime.now().isoformat()
        }
    query_id = str(uuid.uuid4())
    json_save = None
    metadata = load_pdf_metadata()
    pdf_files = list(metadata.keys())
    if not pdf_files:
        print("No indexed PDFs found. Run pdf_indexer.py.")
        json_save = save_to_json(query_id, query_text, "No indexed PDFs found.")
        return json_save
    best_response = None
    best_score = -1
    best_page_number = None
    best_file_path = None
    for file_path in pdf_files:
        vector_index = load_vector_index(file_path, embed_model)
        if not vector_index:
            continue
        response, page_number, score = query_document(vector_index, query_text, llm, file_path)
        if response and page_number and score is not None:
            if score > best_score:
                best_score = score
                best_response = response
                best_page_number = page_number
                best_file_path = file_path
    if best_response:
        img_path = extract_and_highlight(best_file_path, best_page_number, query_text, query_id, IMAGE_DIR)
        audio_path = generate_audio(best_response, query_id, AUDIO_DIR)
        json_save = save_to_json(query_id, query_text, best_response, best_page_number, img_path, audio_path, best_file_path)
        cache_key = query_text.lower()
        QUERY_CACHE[cache_key] = {
            "response": best_response,
            "page_number": best_page_number,
            "image_path": img_path,
            "audio_path": audio_path,
            "pdf_path": best_file_path
        }
        return json_save
    json_save = save_to_json(query_id, query_text, "No relevant answer found.")
    return json_save

def main_method(query_text):
    nest_asyncio.apply()
    hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not hf_token:
        raise ValueError("Missing HUGGINGFACE_API_TOKEN")
    OUTPUT_DIR = "/tmp/outputs"  # Use temporary directory
    IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
    AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = HuggingFaceInferenceAPI(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        contextWindow=32768,
        maxTokens=1024,
        temperature=0.05,
        topP=0.9,
        frequencyPenalty=0.5,
        presencePenalty=0.5,
        token=hf_token
    )
    response = process_query(query_text, embed_model, llm, IMAGE_DIR, AUDIO_DIR)
    return response

app = FastAPI()

@app.post("/query")
async def handle_query(value: str):
    if not value.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        query = value
        response = main_method(query)
        page_list = multiple_image(query)
        if page_list:
            page = ",".join(str(item) for item in page_list).replace('"', '')
            response['page_number'] = page
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
