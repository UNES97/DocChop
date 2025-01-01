import os 
import chromadb
from chromadb.config import Settings 

""" DuckDB (a high-performance in-process database) combined with Parquet files (a columnar storage file format optimized for large-scale data). """
CHROMA_SETTINGS = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory='db',
    anonymized_telemetry=False
)