use std::io::{BufRead, BufReader, Read};

use serde::{Deserialize, Serialize};

use crate::search_engine::{Document, SearchEngine};
use anyhow::Result;

#[derive(Serialize, Deserialize)]
struct DocumentToImport {
    id: String,
    title: Option<String>,
    text: String,
    #[serde(rename = "emb")]
    embedding: Vec<f32>,
}

pub fn import_jsonl(read: impl Read, search_engine: &mut SearchEngine) -> Result<()> {
    let read = BufReader::with_capacity(16 * 1024 * 1024, read);
    let mut batch = vec![];
    for line in read.lines() {
        let line = line?;
        let doc: DocumentToImport = serde_json::from_str(&line)?;
        if doc.embedding.len() as u64 != search_engine.embedding_len() {
            return Err(anyhow::anyhow!(
                "Document {} embedding length {} does not match search engine embedding length {}",
                doc.id,
                doc.embedding.len(),
                search_engine.embedding_len()
            ));
        }
        let mut text = match doc.title {
            Some(title) => format!("{}\n\n{}", title, doc.text),
            None => doc.text,
        };
        if text.contains('\0') {
            text = text.replace('\0', " ");
        }
        let doc = Document {
            id: doc.id,
            text,
            embedding: doc.embedding,
        };
        batch.push(doc);
        if batch.len() >= 10000 {
            search_engine.add_batch(&batch)?;
            batch.clear();
        }
    }
    search_engine.add_batch(&batch)?;
    Ok(())
}
