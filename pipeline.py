
from pathlib import Path

import kfp
from kfp import dsl
from chain import download_documents, transform_to_vectors


@dsl.component
def pipeline_extract(source_repo: str, source_ref: str, source_prefix: str, data_path: str):
    download_documents(source_repo, source_ref, source_prefix, Path(data_path))


@dsl.component
def pipeline_load(data_path: str, index_path: str):
    transform_to_vectors(Path(data_path), Path(index_path))


@dsl.pipeline
def rag_pipeline(source_repo: str, source_ref: str, source_prefix: str, index_path: str):
    pipeline_extract(source_repo=source_repo, source_ref=source_ref, source_prefix=source_prefix, data_path='books')
    pipeline_load(data_path='books', index_path=index_path)


kfp_client = kfp.Client()
kfp_client.create_run_from_pipeline_func(
    rag_pipeline,
    arguments={
        "source_repo": "peter-pan-data",  # our lakeFS repository
        "source_ref": "v2-prod",          # an immutable lakeFS Tag!
        "source_prefix": "data/books/",   # path in repo
        "index_path": "index",            # vector DB location on shared storage
    }
)
