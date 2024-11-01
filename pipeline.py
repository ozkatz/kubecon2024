
from pathlib import Path

from kfp import dsl
from chain import download_documents, transform_to_vectors


@dsl.component
def pipeline_extract(source_bucket: str, source_prefix: str, data_path: str):
    download_documents(source_bucket, source_prefix, Path(data_path))


@dsl.component
def pipeline_load(data_path: str, index_path: str):
    transform_to_vectors(Path(data_path), Path(index_path))


@dsl.pipeline
def rag_pipeline(source_bucket: str, source_prefix: str, index_path: str):
    pipeline_extract(source_bucket=source_bucket, source_prefix=source_prefix, data_path='books')
    pipeline_load(data_path='books', index_path=index_path)
