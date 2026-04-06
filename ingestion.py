# from rich.traceback import install
# install()

import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
# from langchain_chroma import Chroma
from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


embeddings =  OllamaEmbeddings(
                model=os.environ['OLLAMA_EMBEDDING_MODEL'],
                base_url=os.environ['OLLAMA_BASE_URL'],
            )
# OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     show_progress_bar=False,
#     chunk_size=50,
#     retry_min_seconds=10,
# )

# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)


tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asynchronously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"📚 VectorStore Indexing: Preparing to add {len(documents)} chunks to vector store",
        Colors.DARKCYAN,
    )

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches of {batch_size} chunks each"
    )

    # Each task gets its own vectorstore instance to avoid shared HTTP session conflicts
    async def add_batch(batch: List[Document], batch_num: int):
        vs = PineconeVectorStore(
            index_name=os.environ["INDEX_NAME"], embedding=embeddings
        )
        try:
            await vs.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} records)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

    # Process all batches concurrently
    
    import time
    
    start = time.perf_counter()

    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.2f} seconds")


    # results = []
    # for i, batch in enumerate(batches):
    #     res = await add_batch(batch, i + 1)
    #     results.append(res)

    # Count successful batches
    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
        )


async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🔍 TavilyCrawl: Starting to crawl documentation from https://docs.langchain.com/oss/python/",
        Colors.PURPLE,
    )

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://docs.langchain.com/oss/python/",
            "extract_depth": "advanced",
            "max_depth": 1,
        }
    )

    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )

    all_docs = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_docs.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_docs)} pages with 2000 chunk size and 100 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    splitted_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents"
    )

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=100)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Documents extracted: {len(all_docs)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")

if __name__ == "__main__":
    asyncio.run(main())