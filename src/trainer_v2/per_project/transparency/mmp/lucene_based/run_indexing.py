import itertools
import sys

import lucene
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from java.nio.file import Paths

from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader, IndexOptions
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import ByteBuffersDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.util import Version
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_collection


def run_indexing(
        index_dir,
        doc_iter: Iterable[tuple[int, str]],
        analyzer,
):
    # Initialize Lucene and JVM

    # Create a RAMDirectory to hold the index in memory
    directory = ByteBuffersDirectory()
    directory = FSDirectory.open(Paths.get(index_dir))

    # Create a standard analyzer

    # Create an index writer configuration
    config = IndexWriterConfig(analyzer)

    # Create an index writer
    writer = IndexWriter(directory, config)

    # Define a document type with a text field
    textFieldType = FieldType()
    textFieldType.setStored(True)
    textFieldType.setTokenized(True)
    textFieldType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    id_field = FieldType()
    id_field.setStored(True)
    id_field.setTokenized(False)
    id_field.setIndexOptions(IndexOptions.DOCS_AND_FREQS)

    # Add documents to the index
    docs = ["define school", "What is your work", "The Works"]
    for doc_id, content in doc_iter:
        doc = Document()
        doc.add(Field("contents", content, textFieldType))
        doc.add(Field("id", doc_id, id_field))
        writer.addDocument(doc)

    # Commit and close the writer
    writer.commit()
    writer.close()


if __name__ == '__main__':
    itr = load_msmarco_collection()
    itr = itertools.islice(itr, 1000)
    
    lucene.initVM()
    analyzer = StandardAnalyzer()
    run_indexing(sys.argv[1], itr, analyzer)
    # en_analyzer = EnglishAnalyzer()
