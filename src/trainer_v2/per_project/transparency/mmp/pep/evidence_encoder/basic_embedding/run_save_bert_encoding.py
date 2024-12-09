import sys

from cpath import output_path
from misc_lib import path_join

from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.basic_embedding.bert_encodding_common import \
    process_batch

if __name__ == "__main__":
    # Example usage
    job_no = int(sys.argv[1])  # Change this to process different batches
    process_batch(
        job_no,
        "pooler_output",
        file_path=path_join(output_path, "mmp", "bt2_df.tsv"),
        output_file=path_join(output_path, "mmp", "bert_encoding_pooled", str(job_no)),
    )
