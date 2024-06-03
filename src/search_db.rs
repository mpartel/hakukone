use std::{
    collections::{hash_map::Entry, HashMap},
    path::Path,
};

use anyhow::{anyhow, Result};
use rocksdb::{
    AsColumnFamilyRef, Options as RocksDBOptions, Transaction, TransactionDB, TransactionDBOptions,
};
use serde::{Deserialize, Serialize};

use crate::search_engine::SummarizedDocument;

/// A key-value database used by SearchEngine to store some indices.
pub struct SearchDB {
    db: TransactionDB,
    state: SearchDBState,
}

struct SearchDBState {
    last_committed_tx_num: u64,
    transaction_pending: bool,
}

const COLUMN_FAMILIES: &[&str] = &[
    "lemmas",          // Lemma -> LemmaEntry JSON
    "docs",            // Doc index -> IndexedDoc
    "doc_id_to_index", // Doc ID -> Doc index
];

impl SearchDB {
    pub fn open(path: &Path, last_committed_tx_num: u64) -> Result<SearchDB> {
        let mut rocks_options = RocksDBOptions::default();
        rocks_options.create_if_missing(true);
        rocks_options.create_missing_column_families(true);
        rocks_options.set_allow_mmap_reads(true);
        rocks_options.set_allow_mmap_writes(true);
        rocks_options.set_soft_pending_compaction_bytes_limit(128 * 1024 * 1024);
        rocks_options.set_hard_pending_compaction_bytes_limit(4 * 1024 * 1024 * 1024);
        let tx_options = TransactionDBOptions::default();
        let db = TransactionDB::open_cf(&rocks_options, &tx_options, path, COLUMN_FAMILIES)
            .map_err(|e| anyhow::anyhow!("Failed to open BM25Index DB at '{:?}': {}", path, e))?;
        for tx in db.prepared_transactions() {
            match tx.get_name() {
                Some(bytes) => {
                    if bytes.len() != 8 {
                        return Err(anyhow::anyhow!("Invalid transaction name length"));
                    }
                    let tx_num = u64::from_le_bytes(bytes.try_into().unwrap());
                    if tx_num > last_committed_tx_num {
                        tx.rollback()?;
                    } else {
                        tx.commit()?;
                    }
                }
                None => return Err(anyhow::anyhow!("prepared transaction has no name")),
            }
        }
        Ok(SearchDB {
            db,
            state: SearchDBState {
                last_committed_tx_num,
                transaction_pending: false,
            },
        })
    }

    pub fn transaction(&mut self) -> Result<SearchDBTransaction> {
        if self.state.transaction_pending {
            // For recoverability via last_committed_tx_num, we only allow one transaction at a time.
            return Err(anyhow::anyhow!("transaction already pending"));
        }
        let tx = self.db.transaction();
        let tx_num = self.state.last_committed_tx_num + 1;
        self.state.transaction_pending = true;
        tx.set_name(&tx_num.to_le_bytes())?;
        Ok(SearchDBTransaction {
            active: Some(ActiveTransaction {
                db_state: &mut self.state,
                db: &self.db,
                tx,
                tx_num,
            }),
        })
    }

    pub fn get_lemma_entries(&self, lemma: &str) -> impl Iterator<Item = Result<LemmaEntry>> + '_ {
        let lemmas_cf = self
            .db
            .cf_handle("lemmas")
            .expect("column family 'lemmas' not found");

        let mut prefix = Vec::with_capacity(lemma.as_bytes().len() + 1);
        prefix.extend_from_slice(lemma.as_bytes());
        prefix.push(0x00);
        self.prefix_only_iterator_cf(lemmas_cf, prefix).map(|item| {
            let (_, entry) = item?;
            let entry = serde_json::from_slice::<LemmaEntry>(&entry)?;
            Ok(entry)
        })
    }

    /// `prefix_iterator_cf` continues on to keys that no longer have the prefix.
    /// This iterator does not.
    fn prefix_only_iterator_cf<'a, P: AsRef<[u8]> + 'a>(
        &'a self,
        cf: &'a impl AsColumnFamilyRef,
        prefix: P,
    ) -> impl Iterator<Item = std::result::Result<(Box<[u8]>, Box<[u8]>), rocksdb::Error>> + 'a
    {
        let mut error_seen = false;
        self.db
            .prefix_iterator_cf(cf, &prefix)
            .take_while(move |item| match item {
                Ok((k, _)) => k.starts_with(prefix.as_ref()),
                Err(_) if error_seen => {
                    error_seen = true;
                    true
                }
                Err(_) => false,
            })
    }

    pub fn top_lemmas(&self, limit: usize) -> Result<Vec<(String, u64)>> {
        let lemmas_cf = self
            .db
            .cf_handle("lemmas")
            .expect("column family 'lemmas' not found");

        let mut map = HashMap::<String, u64>::new();
        for item in self.db.iterator_cf(lemmas_cf, rocksdb::IteratorMode::Start) {
            let (k, _) = item?;
            let lemma: &[u8] = k.split(|&b| b == 0x00).next().unwrap();
            let lemma = std::str::from_utf8(lemma)?;
            match map.get_mut(lemma) {
                Some(count) => {
                    *count += 1;
                }
                None => {
                    map.insert(lemma.to_string(), 1);
                }
            };
        }
        let mut result = Vec::from_iter(map.into_iter());
        result.sort_by(|(_, c1), (_, c2)| c2.cmp(c1));
        result.truncate(limit);
        Ok(result)
    }

    pub fn doc_id_to_index(&self, doc_id: &str) -> Result<Option<u64>> {
        let doc_id_to_index_cf = self
            .db
            .cf_handle("doc_id_to_index")
            .expect("column family 'doc_id_to_index' not found");
        if let Some(bytes) = self.db.get_cf(doc_id_to_index_cf, doc_id.as_bytes())? {
            assert_eq!(bytes.len(), 8);
            Ok(Some(u64::from_le_bytes(bytes.try_into().unwrap())))
        } else {
            Ok(None)
        }
    }

    pub fn doc_id_prefix_to_indices<'a>(
        &'a self,
        doc_id_prefix: &'a str,
    ) -> impl Iterator<Item = Result<u64>> + 'a {
        let doc_id_to_index_cf = self
            .db
            .cf_handle("doc_id_to_index")
            .expect("column family 'doc_id_to_index' not found");
        let doc_id_prefix = doc_id_prefix.as_bytes();
        self.prefix_only_iterator_cf(doc_id_to_index_cf, doc_id_prefix)
            .map(|result| {
                let (_, bytes) = result?;
                assert_eq!(bytes.len(), 8);
                Ok(u64::from_le_bytes(bytes.as_ref().try_into().unwrap()))
            })
    }

    pub fn doc_index_to_id(&self, doc_index: u64) -> Result<Option<String>> {
        Ok(self.get_doc(doc_index)?.map(|doc| doc.id))
    }

    pub fn get_doc(&self, doc_index: u64) -> Result<Option<IndexedDoc>> {
        let docs_cf = self
            .db
            .cf_handle("docs")
            .expect("column family 'docs' not found");
        if let Some(bytes) = self.db.get_cf(docs_cf, doc_index.to_le_bytes())? {
            let doc: IndexedDoc = serde_json::from_slice(&bytes)?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }

    pub fn get_doc_without_id(&self, doc_index: u64) -> Result<Option<IndexedDocWithoutId>> {
        let docs_cf = self
            .db
            .cf_handle("docs")
            .expect("column family 'docs' not found");
        if let Some(bytes) = self.db.get_cf(docs_cf, doc_index.to_le_bytes())? {
            let doc: IndexedDocWithoutId = serde_json::from_slice(&bytes)?;
            Ok(Some(doc))
        } else {
            Ok(None)
        }
    }
}

pub struct AddBatchResult {
    pub additional_token_count: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct LemmaEntry {
    pub doc_index: u64,
    pub freq: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct IndexedDoc {
    pub id: String,
    #[serde(rename = "wc")]
    pub word_count: u64,
    #[serde(rename = "del")]
    pub deleted: bool,
}

#[derive(Deserialize, Debug)]
pub struct IndexedDocWithoutId {
    #[serde(rename = "wc")]
    pub word_count: u64,
    #[serde(rename = "del")]
    pub deleted: bool,
}

pub struct SearchDBTransaction<'a> {
    active: Option<ActiveTransaction<'a>>,
}

struct ActiveTransaction<'a> {
    db_state: &'a mut SearchDBState,
    db: &'a TransactionDB,
    tx: Transaction<'a, TransactionDB>,
    tx_num: u64,
}

impl<'a> SearchDBTransaction<'a> {
    pub fn add_batch(&mut self, batch: &[SummarizedDocument]) -> Result<AddBatchResult> {
        let active = self.active.as_mut().unwrap();
        let lemmas_cf = active
            .db
            .cf_handle("lemmas")
            .expect("column family 'lemmas' not found");
        let docs_cf = active
            .db
            .cf_handle("docs")
            .expect("column family 'docs' not found");
        let doc_id_to_index = active
            .db
            .cf_handle("doc_id_to_index")
            .expect("column family 'doc_id_to_index' not found");

        let mut additional_token_count: u64 = 0;
        for doc in batch {
            let mut lemmas_for_this_doc = HashMap::<Vec<u8>, LemmaEntry>::new();

            for token in doc.tokens.iter() {
                for lemma in token.lemmas.iter() {
                    if lemma.as_bytes().contains(&0x00) {
                        return Err(anyhow!("lemma contains null byte"));
                    }

                    let mut key = Vec::with_capacity(lemma.as_bytes().len() + 9);
                    key.extend_from_slice(lemma.as_bytes());
                    key.push(0);
                    key.extend_from_slice(&doc.index.to_le_bytes());
                    match lemmas_for_this_doc.entry(key) {
                        Entry::Occupied(mut e) => {
                            assert_eq!(e.get().doc_index, doc.index);
                            e.get_mut().freq += 1;
                        }
                        Entry::Vacant(e) => {
                            e.insert(LemmaEntry {
                                doc_index: doc.index,
                                freq: 1,
                            });
                        }
                    }
                }
            }

            for (k, v) in lemmas_for_this_doc {
                active.tx.put_cf(lemmas_cf, k, serde_json::to_vec(&v)?)?;
            }

            let indexed_doc_json = serde_json::to_vec(&IndexedDoc {
                id: doc.id.clone(),
                word_count: doc.tokens.len() as u64,
                deleted: false,
            })?;
            active
                .tx
                .put_cf(docs_cf, doc.index.to_le_bytes(), indexed_doc_json)?;
            active
                .tx
                .put_cf(doc_id_to_index, &doc.id, doc.index.to_le_bytes())?;

            additional_token_count += doc.tokens.len() as u64;
        }

        Ok(AddBatchResult {
            additional_token_count,
        })
    }

    pub fn mark_deleted(&mut self, doc_index: u64) -> Result<Option<DeleteResult>> {
        let active = self.active.as_mut().unwrap();
        let docs_cf = active
            .db
            .cf_handle("docs")
            .expect("column family 'docs' not found");
        if let Some(doc_json) = active.tx.get_cf(docs_cf, doc_index.to_le_bytes())? {
            let mut doc: IndexedDoc = serde_json::from_slice(&doc_json)?;
            doc.deleted = true;
            let doc_json = serde_json::to_vec(&doc)?;
            active
                .tx
                .put_cf(docs_cf, doc_index.to_le_bytes(), doc_json)?;
            Ok(Some(DeleteResult {
                word_count: doc.word_count,
            }))
        } else {
            Ok(None)
        }
    }

    fn rollback(&mut self) {
        let active = self.active.as_mut().unwrap();
        match active.tx.rollback() {
            Ok(_) => {}
            Err(e) => {
                panic!("failed to rollback transaction: {}", e);
            }
        }
        active.db_state.transaction_pending = false;
    }

    pub fn prepare(mut self) -> Result<SearchDBPreparedTransaction<'a>> {
        let active = self.active.take().unwrap();
        active.tx.prepare()?;
        Ok(SearchDBPreparedTransaction { active })
    }
}

impl<'a> Drop for SearchDBTransaction<'a> {
    fn drop(&mut self) {
        if self.active.is_some() {
            self.rollback();
        }
    }
}

pub struct SearchDBPreparedTransaction<'a> {
    active: ActiveTransaction<'a>,
}

impl<'a> SearchDBPreparedTransaction<'a> {
    pub fn tx_num(&self) -> u64 {
        self.active.tx_num
    }

    pub fn commit(self) -> Result<()> {
        let active = self.active;
        assert_eq!(active.tx_num, active.db_state.last_committed_tx_num + 1);
        assert!(active.db_state.transaction_pending);
        active
            .tx
            .commit()
            .map_err(|e| anyhow::anyhow!("failed to commit transaction: {}", e))?;
        active.db_state.last_committed_tx_num = active.tx_num;
        active.db_state.transaction_pending = false;
        Ok(())
    }
}

pub struct DeleteResult {
    pub word_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voikko_tokenizer::Token;

    macro_rules! test_dir {
        () => {
            crate::testing::clear_and_get_test_dir(function_name!())
        };
    }

    fn add_test_data(tx: &mut SearchDBTransaction) {
        let batch = vec![
            SummarizedDocument {
                id: "doc1".to_string(),
                index: 0,
                tokens: vec![
                    Token {
                        lemmas: vec!["kissa".to_string()],
                        byte_offset: 0,
                    },
                    Token {
                        lemmas: vec!["koira".to_string()],
                        byte_offset: 6,
                    },
                    Token {
                        lemmas: vec!["kissa".to_string()],
                        byte_offset: 12,
                    },
                ],
            },
            SummarizedDocument {
                id: "doc2".to_string(),
                index: 1,
                tokens: vec![Token {
                    lemmas: vec!["kissa".to_string()],
                    byte_offset: 0,
                }],
            },
        ];
        tx.add_batch(&batch).expect("failed to add batch");
    }

    #[test]
    fn basic_init() {
        let dir = test_dir!();
        SearchDB::open(&dir, 0).expect("failed to open SearchDB");
    }

    #[test]
    fn lemma_search() {
        let dir = test_dir!();
        let mut db = SearchDB::open(&dir, 0).expect("failed to open SearchDB");
        let mut tx = db.transaction().expect("failed to create transaction");
        add_test_data(&mut tx);
        let tx = tx.prepare().expect("failed to prepare transaction");
        tx.commit().expect("failed to commit transaction");

        let lemmas = db
            .get_lemma_entries("kissa")
            .map(|e| e.unwrap())
            .collect::<Vec<LemmaEntry>>();
        assert_eq!(lemmas.len(), 2);
        assert_eq!(lemmas[0].doc_index, 0);
        assert_eq!(lemmas[0].freq, 2);
        assert_eq!(lemmas[1].doc_index, 1);
        assert_eq!(lemmas[1].freq, 1);
    }

    #[test]
    fn lookup_by_id() {
        let dir = test_dir!();
        let mut db = SearchDB::open(&dir, 0).expect("failed to open SearchDB");
        let mut tx = db.transaction().expect("failed to create transaction");
        add_test_data(&mut tx);
        let tx = tx.prepare().expect("failed to prepare transaction");
        tx.commit().expect("failed to commit transaction");

        assert_eq!(db.doc_id_to_index("doc1").unwrap(), Some(0));
        assert_eq!(db.doc_id_to_index("doc2").unwrap(), Some(1));
        assert_eq!(db.doc_id_to_index("doc3").unwrap(), None);
    }

    #[test]
    fn lookup_by_index() {
        let dir = test_dir!();
        let mut db = SearchDB::open(&dir, 0).expect("failed to open SearchDB");
        let mut tx = db.transaction().expect("failed to create transaction");
        add_test_data(&mut tx);
        let tx = tx.prepare().expect("failed to prepare transaction");
        tx.commit().expect("failed to commit transaction");

        assert!(matches!(
            db.get_doc(0).unwrap(),
            Some(IndexedDoc {
                id,
                word_count: 3,
                deleted: false
            }) if id == "doc1"
        ));
        assert!(matches!(
            db.get_doc(1).unwrap(),
            Some(IndexedDoc {
                id,
                word_count: 1,
                deleted: false
            }) if id == "doc2"
        ));
        assert!(matches!(db.get_doc(2).unwrap(), None));
    }
}
