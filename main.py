import os
from dotenv import load_dotenv
import typer
import pandas as pd
from rich import print
from pypdf import PdfReader
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity


app = typer.Typer()
FILTER_TEXT_LENGTH = 30
EMBEDDING_MODEL = "text-embedding-ada-002"

def parse_paper(path):
    print("Parsing paper")
    reader = PdfReader(path)
    n_pages = len(reader.pages)
    print(f"Total number of pages: {n_pages}")
    paper_text = []
    for i in range(n_pages):
        page = reader.pages[i]
        page_text = []
        
        def visitor_body(text, cm, tm, fontDict, fontSize):
            x = tm[4]
            y = tm[5]
            # ignore header and footer
            if (y > 50 and y < 720) and (len(text.strip()) > 1):
                page_text.append({
                    "fontsize": fontSize,
                    "text": text.strip().replace("\x03", ""),
                    "x": x,
                    "y": y
                })
        
        _ = page.extract_text(visitor_text=visitor_body)
        
        blob_font_size = None
        blob_text = ""
        processed_text = []
        
        for t in page_text:
            if t["fontsize"] == blob_font_size:
                blob_text += f" {t['text']}"
            else:
                if blob_font_size is not None and len(blob_text) > 1:
                    processed_text.append({
                        "fontsize": blob_font_size,
                        "text": blob_text,
                        "page": i
                    })
                blob_font_size = t["fontsize"]
                blob_text = t["text"]
        paper_text += processed_text
    return paper_text


def search_paper(df, query):
    query_embedding = get_embedding(
        query,
        engine=EMBEDDING_MODEL
    )
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x, query_embedding))
    results = df.sort_values("similarity", ascending=False)
    return results.iloc[0]["text"]


@app.command()
def main(filename: str):
    if not os.path.exists(filename):
        print(f"[red]The file {filename} does not exist!")
        raise typer.Exit(1)
    load_dotenv("./")
    print(os.environ)
    if not os.getenv("OPENAI_API_KEY"):
        print(f"[red]OPENAI_API_KEY environment variable not found!")
        raise typer.Exit(1)
    openai.api_key = os.getenv("OPENAI_API_KEY")    
    paper_text = parse_paper(filename)
    # Get rid of short text
    filtered_paper_text = []
    for row in paper_text:
        if len(row["text"]) < FILTER_TEXT_LENGTH:
            continue
        filtered_paper_text.append(row)
    df = pd.DataFrame(filtered_paper_text)
    embeddings = df.text.apply([lambda x: get_embedding(x, engine=EMBEDDING_MODEL)])
    df["embeddings"] = embeddings
    while True:
        user_query = typer.prompt("Enter your query. Enter nothing to quit.", default="", show_default=False)
        if user_query == "":
            raise typer.Exit()
        print(search_paper(df, user_query))


if __name__ == "__main__":
    app()