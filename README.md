# Hakukone

Hakukone ("search engine" in Finnish) is a simple full-text search backend specifically for the Finnish language.

Given a set of Finnish text documents, Hakukone provides a web service with the following functions:

- Keyword search using [Voikko](https://voikko.puimula.org/) for lemmatization
  and [BM25+](https://en.wikipedia.org/wiki/Okapi_BM25) for ranking
- Embedding-based search using cosine distance for ranking
- Adding a batch of new documents
- Marking a document as soft-deleted
- Compacting i.e. rewriting the database in the background while dropping soft-deleted documents

While this should be regarded as a [hobby project](https://en.wikipedia.org/wiki/Not_invented_here)
that pales in comparison to established full text search systems
like Solr/Elasticsearch/Meilisearch/Tantivy, it has a couple of advantages:

- It uses Voikko for lemmatization, whereas other systems seem to use more coarse stemming rules.
- It supports both keyword-based search and embedding-based ("semantic") search. Many alternatives don't provide both.
- It's small and simple, and thus relatively easy to understand, modify, deploy and maintain.

## Goals and scope

This project aims to do a decent job of full-text search in Finnish.
Therefore it may develop additional tuning for the Finnish language e.g. in the way that Voikko is applied.

Otherwise the project is pretty much feature complete. For instance, it does **not** aim to:

- provide a UI
- combine keyword-based and embedding-based search results
  - A frontend can use e.g. [RRF](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) to do that.
- structured documents (fields, schemas, ...)
- reach state-of-the-art performance for large datasets (see below)

## Performance

Performance is good enough for small to medium datasets.

With the Finnish Wikipedia dataset (~880k documents, ~100M words as of 2024-05-01),
using a Ryzen 5900X with Precision Boost off and 4.2GHz all core max:

- Embedding-based search with 1536-dimensional embeddings takes about 150ms with all 24 threads,
  or 1.1 seconds with a single thread.
  - Scaling: `O(documents_in_database * embedding_length)`
- Keyword search takes about 20-30ms per uncommon search word,
  but very common words like "ja" can take around 250ms (or 1 second with a single thread).
  - Scaling: `O(query_words * documents_matched)`

Scaling could be improved with indices like HNSW, but for my simple use cases I prefer the simplicity
and predictability of the current setup: no need to tune index parameters for speed/accuracy etc.
Horizontal scaling is also [possible](#scalability-and-reliability).

# How to use

## Import data

You can do an offline data import on the command line as shown here, or you can use the [API](#api).

Prepare a file with one JSON object per line.
The objects should have the following fields:

- `id` (string)
- `text` (string)
- `emb` (list of numbers) - the embeddings, e.g. [from OpenAI](https://platform.openai.com/docs/guides/embeddings).

Pipe the file to the following command to import the data:

```bash
cat data.jsonl | cargo run -- import-jsonl --data-dir ./data --embedding-len 1536
```

## Run server

```bash
cargo run -- serve --host 127.0.0.1 --data-dir ./data --embedding-len 1536
```

Test:

```bash
curl \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{"query": "kissan karvat", "limit": 10}' \
  http://localhost:3888/api/search/by-text
```

See [API](#api) below for further instructions.

## Deployment

There's a [Dockerfile](Dockerfile):

```bash
docker build -t hakukone:latest .
docker run -d -p 3888:3888 -v ./data:/app/data hakukone:latest serve --embedding-len 1536
```

### Security

While this project uses very little unsafe Rust, it has some dependencies written in C/C++.
Most importantly, text queries are passed to Voikko's tokenizer and analyzer.
While Voikko is mature software, input validation and strong sandboxing may still be wise.

### Scalability and reliability

Search latency and throughput can be improved with additional cores.

Sharding and replication are possible by running multiple instances.

In any deployment, you should keep all document insertions in a reliable queue
in your main application database so they are not lost if a search server is temporarily unavailable.

### Disaster recovery

Ideally your main application should have a way to re-insert all documents to the search server.

Alternatively, you can back up the file `data/live/originals.jsonl` while the server is running.
It can then be [imported](#import-data) to rebuild the search index.
The file is append-only while no compactions are performed, which may be useful for incremental backups.

# API

All requests are POST with a JSON body.

### `/api/search/by-text`

Does keyword-based search using BM25+ for ranking.

#### Request

- `query`: The search keywords separated by whitespace. (No fancy syntax such as boolean operations or quoted phrases are supported.)
- `limit`: The number of results to return.

#### Response

- `results`: A list of objects with the following fields:
  - `doc_id` (string)
  - `score` (number) - BM25+ score: higher is better

### `/api/search/by-embedding`

Does embedding-based search using cosine distance for ranking.

#### Request

- `embedding`: The embedding - a list of numbers.
- `limit`: The number of results to return.

#### Response

- `results`: A list of objects with the following fields:
  - `doc_id` (string)
  - `score` (number) - Cosine distance score: lower is better

### `/api/add-batch`

Indexes new documents to the database.

Idempotence: If any of the provided document IDs are already in the database (even if marked for deletion),
they will **not** be overwritten, but their IDs are returned in the response.

To work around the inability to efficiently overwrite documents,
you can design your document IDs to be something like `<internal-id>@<version-or-timestamp>`.

This request will fail with a HTTP 409 (Conflict) if there's an ongoing compaction.
The frontend should retry later in this case.

#### Request

- `docs`: A list of objects with the following fields:

  - `id` (string)
  - `text` (string)
  - `emb` (the embeddding - a list of numbers)

#### Response

- `existing_doc_ids` (list of strings) - document IDs that already existed in the database and were this ignored

### `/api/delete`

Marks a document as soft-deleted. It will no longer appear in search results,
and it will be removed from the database on compaction.

This request will fail with a HTTP 409 (Conflict) if there's an ongoing compaction.
The frontend should retry later in this case.

Idempotence: the request will succeed even if the document does not exist or is already
marked as soft-deleted.

#### Request

- `doc_id` (string)

#### Response

Empty object.

### `/api/delete/by-prefix`

Marks all documents whose document ID starts with the given prefix.

This request will fail with a HTTP 409 (Conflict) if there's an ongoing compaction.
The frontend should retry later in this case.

Idempotence: the request will succeed even if some or all of the documents do not exist or are already
marked as soft-deleted.

#### Request

- `doc_id_prefix` (string)

#### Response

Empty object.

### `/api/compact`

Rewrites the database in the background and swaps it in for the live database when complete.

Only one compaction can run at a time.
This request will fail with a HTTP 409 (Conflict) if there's already an ongoing compaction.

Currently this request blocks until the compaction is complete.
This may take a long time if the dataset is large.

#### Request

Empty object.

#### Response

Empty object.

### `/api/clear-database`

Completely clears the database.

This request will fail with a HTTP 409 (Conflict) if there's an ongoing compaction.
The frontend should retry later in this case.

#### Request

Empty object.

#### Response

Empty object.

# Built with

- [Voikko](https://voikko.puimula.org/) for stemming/lemmatization
- [RocksDB](https://rocksdb.org/) for storing the reverse index (word lemmas -> documents)
- [BM25+](https://en.wikipedia.org/wiki/Okapi_BM25) for ranking in keyword-based search
- [simsimd](https://github.com/ashvardanian/SimSIMD) and [Rayon](https://github.com/rayon-rs/rayon)
  for searching through the embeddings (no index at the moment)
- [Actix Web](https://actix.rs/) for serving the HTTP API

# License

[GPLv3](LICENSE.txt) or later (same as Voikko)
