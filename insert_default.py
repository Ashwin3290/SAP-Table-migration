import os
import json
from pymongo import MongoClient

if __name__ == '__main__':
    current_file_path = os.path.abspath(__file__)
    curr_dir = os.path.dirname(current_file_path)

    # Connect to MongoDB without authentication
    client = MongoClient('localhost', 27017).table_llm

    with open(f'{curr_dir}/default_data/docs/table-merge-200-final.jsonl', 'r') as fp:
        docs = [json.loads(line) for line in fp.readlines()]
    client.table_merge.insert_many(docs)

    with open(f'{curr_dir}/default_data/docs/table-op-1000-final.jsonl', 'r') as fp:
        docs = [json.loads(line) for line in fp.readlines()]
    client.table_op.insert_many(docs)

    with open(f'{curr_dir}/default_data/docs/wtq-633-final.jsonl', 'r') as fp:
        docs = [json.loads(line) for line in fp.readlines()]
    client.wtq.insert_many(docs)