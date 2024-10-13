import tempfile
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
import os
from dotenv import load_dotenv
import zipfile
from utils import process_document, add_document_to_db, search_in_db
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from chromadb.utils import embedding_functions
import chromadb
from chromadb.config import Settings

load_dotenv()

app = FastAPI()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CHROMA_HOST = os.getenv("CHROMA_HOST")
CHROMA_PORT = os.getenv("CHROMA_PORT")

client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=int(CHROMA_PORT),
    settings=Settings()
)

shared_embedder = HuggingFaceEmbeddings(model_name="deepvk/USER-base")

text_splitter = SemanticChunker(shared_embedder, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=65)

class ChromaEmbeddingFunction:
    def __init__(self, embedder):
        self.embedder = embedder

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.embedder.embed_documents(input)

chroma_embedding_function = ChromaEmbeddingFunction(shared_embedder)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=chroma_embedding_function,
)


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        document_text = process_document(file.filename, content)
        add_document_to_db(document_text, collection, text_splitter)
        return {"filename": file.filename, "status": "Uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_folder")
async def upload_folder(file: UploadFile = File(...)):
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = os.path.join(tmpdir, "uploaded_archive.zip")

            # Save the uploaded file to the temporary directory
            async with aiofiles.open(archive_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)

            # Extract the contents of the archive
            with zipfile.ZipFile(archive_path, 'r') as archive:
                archive.extractall(tmpdir)

            # Process each file in the extracted directory
            for root, _, files in os.walk(tmpdir):
                for name in files:
                    file_path = os.path.join(root, name)
                    
                    # Skip unsupported file types
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in ALLOWED_EXTENSIONS:
                        continue

                    print(f"Processing file: {name}")

                    # Open and read the content of each file
                    async with aiofiles.open(file_path, 'rb') as f:
                        content = await f.read()
                    
                    try:
                        # Process document and add to DB
                        document_text = process_document(name, content)
                        add_document_to_db(document_text, collection, text_splitter)
                    except ValueError as ve:
                        print(f"Error processing file {name}: {ve}")
                        continue

        return JSONResponse(status_code=200, content={"status": "Uploaded all files successfully"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_document(query: str, num: int):
    try:
        results = search_in_db(query, collection, num)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
