from pathlib import Path
from typing import Dict, List

from flask import current_app
from biotrainer.protocols import Protocol
from biotrainer.embedders import get_embedding_service

from biocentral_server.server_management.embedding_database import EmbeddingsDatabaseTriple, EmbeddingsDatabase


def compute_embeddings(embedder_name: str,
                       all_seqs: Dict[str, str],
                       embeddings_out_path: Path,
                       reduce_by_protocol: Protocol,
                       use_half_precision: bool,
                       device) -> List[EmbeddingsDatabaseTriple]:
    embedding_service = get_embedding_service(embedder_name=embedder_name, embeddings_file_path=None,
                                              use_half_precision=use_half_precision, device=device)
    embeddings_file_path = embedding_service.compute_embeddings(input_data=all_seqs, output_dir=embeddings_out_path,
                                                                protocol=reduce_by_protocol, force_output_dir=True)
    computed_embeddings = embedding_service.load_embeddings(embeddings_file_path)
    return EmbeddingsDatabase.unify_seqs_with_embeddings(seqs=all_seqs, embds=computed_embeddings)


def compute_embeddings_and_save_to_db(embedder_name: str,
                                      all_seqs: Dict[str, str],
                                      embeddings_out_path: Path,
                                      reduce_by_protocol: Protocol,
                                      use_half_precision: bool,
                                      device) -> List[EmbeddingsDatabaseTriple]:
    reduce = reduce_by_protocol in Protocol.per_sequence_protocols()
    embeddings_db: EmbeddingsDatabase = current_app.config["EMBEDDINGS_DATABASE"]
    existing_embds_seqs, non_existing_embds_seqs = embeddings_db.filter_existing_embeddings(sequences=all_seqs,
                                                                                            embedder_name=embedder_name,
                                                                                            reduced=reduce)

    if len(non_existing_embds_seqs) > 0:
        computed_embeddings_triples = compute_embeddings(embedder_name=embedder_name,
                                                         all_seqs=non_existing_embds_seqs,
                                                         embeddings_out_path=embeddings_out_path,
                                                         reduce_by_protocol=reduce_by_protocol,
                                                         use_half_precision=use_half_precision,
                                                         device=device)
        embeddings_db.save_embeddings(ids_seqs_embds=computed_embeddings_triples, embedder_name=embedder_name,
                                      reduced=reduce)

    return embeddings_db.get_embeddings(sequences=all_seqs, embedder_name=embedder_name,
                                        reduced=reduce)
