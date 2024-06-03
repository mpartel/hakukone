use std::{
    cmp::Ordering,
    collections::{hash_map::Entry, HashMap, HashSet},
    ffi::CString,
    fs,
    io::{self, BufWriter, Seek, Write},
    mem,
    os::unix::{ffi::OsStrExt, fs::MetadataExt},
    path::{Path, PathBuf},
    slice, thread,
};

use anyhow::{anyhow, Context};
use float_ord::FloatOrd;
use fslock::LockFile;
use log::info;
use mmap_rs::{Mmap, MmapFlags, MmapOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use simsimd::SpatialSimilarity;
use thiserror::Error;

use crate::{
    deref_option::DerefOption,
    files::{read_json_file_if_exists, read_json_lines_from_file_if_exists, replace_json_file},
    machine_type::machine_type,
    search_db::{DeleteResult, SearchDB},
    voikko_tokenizer::{Token, VoikkoTokenizer},
};

pub struct SearchEngine {
    main_record: MainRecord,

    db: DerefOption<SearchDB>,
    embeddings_mmap: Mmap,

    paths: SearchEnginePaths,

    #[allow(dead_code)] // Held for dropping
    process_lock: LockFile,
}

struct SearchEnginePaths {
    data_dir: PathBuf,
    // Holds the main record as JSON
    main: PathBuf,
    // Holds the original document data as JSON lines
    originals: PathBuf,
    // Holds embeddings as raw floats
    embeddings: PathBuf,
    // RocksDB database
    db: PathBuf,
}

#[derive(Serialize, Deserialize, Clone, PartialEq, Debug)]
pub struct Options {
    pub embedding_len: u64,
    pub bm25_k1: f64,
    pub bm25_b: f64,
    pub bm25_delta: f64,
}

impl Default for Options {
    fn default() -> Self {
        Options::default()
    }
}

impl Options {
    pub const fn default() -> Self {
        Self {
            embedding_len: 0,
            bm25_k1: 1.25,
            bm25_b: 0.75,
            bm25_delta: 1.0,
        }
    }
}

#[derive(Error, Debug)]
pub enum SearchEngineError {
    #[error("Document with id '{0}' does not exist")]
    DocIdDoesNotExist(String),
    #[error("Bad settings: {0}")]
    BadSettingsError(String),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub struct AddBatchResult {
    pub existing_doc_ids: Vec<String>,
}

impl From<std::io::Error> for SearchEngineError {
    fn from(e: std::io::Error) -> Self {
        Self::Other(anyhow::Error::from(e))
    }
}

type Result<T> = anyhow::Result<T, SearchEngineError>;

impl SearchEngine {
    pub fn new(dir: &Path, options: Options) -> Result<SearchEngine> {
        options.validate()?;

        fs::create_dir_all(&dir)?;

        let lock_path = dir.join(LOCK_FILE_NAME);

        let paths = SearchEnginePaths {
            data_dir: dir.to_path_buf(),
            main: dir.join(MAIN_FILE_NAME),
            originals: dir.join(ORIGINALS_FILE_NAME),
            embeddings: dir.join(EMBEDDINGS_FILE_NAME),
            db: dir.join(DB_FILE_NAME),
        };

        let machine_type = machine_type().context("getting machine type")?;

        let mut process_lock = LockFile::open(&lock_path)?;
        process_lock.lock_with_pid()?;

        let main_record = match read_json_file_if_exists::<MainRecord>(&paths.main)? {
            Some(main_record) => {
                if main_record.machine_type != machine_type {
                    return Err(anyhow!(
                        "The index was written on a {} machine, but this is a {}. The index must be recreated.",
                        main_record.machine_type,
                        machine_type
                    ).into());
                }
                if main_record.options != options {
                    return Err(anyhow!(
                        "Stored settings differ from given settings. The index must be recreated."
                    )
                    .into());
                }
                main_record
            }
            None => MainRecord {
                machine_type: machine_type.clone(),
                options,
                doc_count: 0,
                deleted_docs: 0,
                total_doc_words: 0,
                originals_file_len: 0,
                embeddings_file_len: 0,
                last_committed_db_tx: 0,
            },
        };

        let mut db: DerefOption<SearchDB> = DerefOption::None;
        Self::revert_file_changes(&paths, &main_record, &mut db)?;
        assert!(db.is_some());

        let embeddings_mmap = Self::create_embeddings_mmap(&main_record, &paths.embeddings)?;

        replace_json_file(&paths.main, &main_record)?;

        Ok(SearchEngine {
            main_record,
            db,
            embeddings_mmap,
            paths,
            process_lock,
        })
    }

    fn check_or_fix_file_len(path: &Path, expected_len: u64) -> Result<()> {
        let actual_len = match fs::metadata(path) {
            Ok(m) => m.size(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // We create the file if it doesn't exist
                fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)?;
                0
            }
            Err(e) => {
                return Err(e)
                    .with_context(|| format!("checking length of file '{:?}'", path))
                    .map_err(|e| e.into())
            }
        };
        if actual_len < expected_len {
            return Err(anyhow!(
                "File '{:?}' is shorter ({} bytes) than it should be ({} bytes)",
                path,
                actual_len,
                expected_len
            )
            .into());
        }
        if actual_len > expected_len {
            info!(
                "Truncating file {:?} to expected length {} from length {}. This may happen when recovering from a crash.",
                path, expected_len, actual_len
            );
            let file = fs::OpenOptions::new().write(true).open(path)?;
            file.set_len(expected_len)?;
        }
        Ok(())
    }

    fn create_embeddings_mmap(main_record: &MainRecord, embeddings_path: &Path) -> Result<Mmap> {
        let embeddings_file = fs::File::open(&embeddings_path)?;

        let mut mapping_size: usize = main_record.embeddings_file_len as usize;
        mapping_size = mapping_size.max(16);
        mapping_size = mapping_size.next_power_of_two();

        unsafe {
            if main_record.embeddings_file_len > 0 {
                Ok(MmapOptions::new(mapping_size)
                    .map_err(|e| anyhow::Error::from(e))?
                    .with_flags(
                        MmapFlags::SHARED
                            | MmapFlags::SEQUENTIAL
                            | MmapFlags::TRANSPARENT_HUGE_PAGES
                            | MmapFlags::NO_CORE_DUMP
                            | MmapFlags::POPULATE,
                    )
                    .with_file(&embeddings_file, 0)
                    .map()
                    .context("memory-mapping embeddings file")?)
            } else {
                // We can't map an empty file, it seems.
                Ok(MmapOptions::new(8)
                    .map_err(|e| anyhow::Error::from(e))?
                    .map()
                    .map_err(|e| anyhow::Error::from(e))?)
            }
        }
    }

    pub fn data_dir(&self) -> &Path {
        &self.paths.data_dir
    }

    pub fn embedding_len(&self) -> u64 {
        self.main_record.options.embedding_len
    }

    pub fn add_batch(&mut self, docs: &[Document]) -> Result<AddBatchResult> {
        if docs.len() == 0 {
            return Ok(AddBatchResult {
                existing_doc_ids: Vec::new(),
            });
        }

        match self.add_batch_preliminary(docs) {
            Ok(prepared_batch) => {
                let add_result = self.apply_add_batch_to_in_memory_datastructures(prepared_batch);
                Ok(add_result)
            }
            Err(e) => {
                Self::revert_file_changes(&self.paths, &self.main_record, &mut self.db)?;
                Err(e)
            }
        }
    }

    // Modifies data files but not the in-memory structures.
    // If this fails, we revert file changes to match the in-memory state.
    fn add_batch_preliminary(
        &mut self,
        docs_with_possible_dupes: &[Document],
    ) -> Result<PreparedAdd> {
        let mut docs = Vec::<&Document>::new();
        let mut existing_doc_ids = Vec::<String>::new();
        {
            let mut seen = HashSet::<&str>::new();
            for doc in docs_with_possible_dupes.iter() {
                if seen.contains(doc.id.as_str()) || self.db.doc_id_to_index(&doc.id)?.is_some() {
                    existing_doc_ids.push(doc.id.clone());
                    continue;
                }
                docs.push(doc);
                seen.insert(&doc.id);
            }
        }

        thread::scope(|s| -> Result<PreparedAdd> {
            let mut main_record = self.main_record.clone();

            let originals_file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.paths.originals)?;
            let embeddings_file = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.paths.embeddings)?;

            let originals_thread = s.spawn(|| -> Result<u64> {
                let mut writer = BufWriter::new(originals_file);
                for doc in docs.iter() {
                    serde_json::to_writer(&mut writer, doc).map_err(|e| anyhow::Error::from(e))?;
                    writer.write(b"\n")?;
                }
                writer.flush()?;
                Ok(writer.stream_position()?)
            });

            let embeddings_thread = s.spawn(|| -> Result<u64> {
                let mut writer = BufWriter::new(embeddings_file);
                for doc in docs.iter() {
                    if doc.embedding.len() as u64 != self.main_record.options.embedding_len {
                        return Err(anyhow!(
                            "Document {} has {} embedding coordinates, {} expected",
                            doc.id,
                            doc.embedding.len(),
                            self.main_record.options.embedding_len
                        )
                        .into());
                    }

                    unsafe {
                        let bytes = slice::from_raw_parts(
                            doc.embedding.as_ptr() as *const u8,
                            doc.embedding.len() * mem::size_of::<f32>(),
                        );
                        writer.write_all(bytes)?;
                    }
                }
                writer.flush()?;
                Ok(writer.stream_position()?)
            });

            let summarized_doc_results: Vec<Result<SummarizedDocument>> = docs
                .par_iter()
                .enumerate()
                .map(|(i, doc)| {
                    let voikko = VoikkoTokenizer::thread_local()?;
                    let doc_index = main_record.doc_count + i as u64;
                    Ok(SummarizedDocument::from_original_document(
                        doc, doc_index, &voikko,
                    ))
                })
                .collect();

            let mut summarized_docs: Vec<SummarizedDocument> = Vec::new();
            for doc_result in summarized_doc_results {
                let doc: SummarizedDocument =
                    doc_result.map_err(|e| SearchEngineError::Other(e.into()))?;
                summarized_docs.push(doc);
            }

            let mut tx = self.db.transaction()?;
            let add_batch_result = tx.add_batch(&summarized_docs)?;

            let new_originals_file_len = originals_thread
                .join()
                .expect("joining originals_file writer")
                .with_context(|| format!("writing to {:?}", self.paths.originals))?;
            let new_embeddings_file_len = embeddings_thread
                .join()
                .expect("joining embeddings_file writer")
                .with_context(|| format!("writing to {:?}", self.paths.embeddings))?;

            let tx = tx.prepare()?;

            main_record.originals_file_len = new_originals_file_len;
            main_record.embeddings_file_len = new_embeddings_file_len;
            main_record.total_doc_words += add_batch_result.additional_token_count;
            main_record.doc_count += docs.len() as u64;
            main_record.last_committed_db_tx = tx.tx_num();

            // If we crash before the main record is updated,
            // the recovery process reverts the prepared transaction.
            // If we crash after, the recovery process commits it.
            replace_json_file(&self.paths.main, &main_record)?;

            tx.commit()?;

            let new_embeddings_mmap =
                Self::create_embeddings_mmap(&main_record, &self.paths.embeddings)?;

            Ok(PreparedAdd {
                updated_main_record: main_record,
                new_embeddings_mmap,
                existing_doc_ids,
            })
        })
    }

    fn apply_add_batch_to_in_memory_datastructures(
        &mut self,
        batch: PreparedAdd,
    ) -> AddBatchResult {
        self.main_record = batch.updated_main_record;
        self.embeddings_mmap = batch.new_embeddings_mmap;
        AddBatchResult {
            existing_doc_ids: batch.existing_doc_ids,
        }
    }

    fn revert_file_changes(
        paths: &SearchEnginePaths,
        main_record: &MainRecord,
        db: &mut DerefOption<SearchDB>,
    ) -> Result<()> {
        // TODO: test forcing an error with a #[cfg(test)] error injection flag.
        // The system search index should still continue to work.
        Self::check_or_fix_file_len(&paths.originals, main_record.originals_file_len)?;
        Self::check_or_fix_file_len(&paths.embeddings, main_record.embeddings_file_len)?;

        *db = DerefOption::None;
        match SearchDB::open(&paths.db, main_record.last_committed_db_tx) {
            Ok(new_db) => *db = DerefOption::Some(new_db),
            Err(e) => {
                // We can't recover from this into a consistent state.
                // Designing a safe API to avoid this potential panic is possible, but not worth the effort.
                panic!("reopening db failed: {}", e);
            }
        }
        Ok(())
    }

    pub fn delete(&mut self, doc_index: u64) -> Result<()> {
        match self.delete_preliminary(doc_index) {
            Ok(Some(delete)) => {
                self.apply_delete_to_in_memory_datastructures(delete);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => {
                Self::revert_file_changes(&self.paths, &self.main_record, &mut self.db)?;
                Err(e)
            }
        }
    }

    fn delete_preliminary(&mut self, doc_index: u64) -> Result<Option<PreparedDelete>> {
        let mut tx = self.db.transaction()?;
        if let Some(DeleteResult { word_count }) = tx.mark_deleted(doc_index)? {
            let tx = tx.prepare()?;

            let mut new_main_record = self.main_record.clone();
            new_main_record.total_doc_words -= word_count;
            new_main_record.deleted_docs += 1;
            new_main_record.last_committed_db_tx = tx.tx_num();
            // We don't decrement doc_count. It's meant to include deleted docs.

            replace_json_file(&self.paths.main, &new_main_record)?;

            tx.commit()?;

            Ok(Some(PreparedDelete { new_main_record }))
        } else {
            Ok(None)
        }
    }

    fn apply_delete_to_in_memory_datastructures(&mut self, delete: PreparedDelete) {
        self.main_record = delete.new_main_record;
    }

    pub fn doc_id_to_index(&self, doc_id: &str) -> Result<Option<u64>> {
        self.db
            .doc_id_to_index(doc_id)
            .map_err(|e| SearchEngineError::Other(e.into()))
    }

    pub fn doc_id_prefix_to_indices<'a>(
        &'a self,
        doc_id_prefix: &'a str,
    ) -> impl Iterator<Item = Result<u64>> + 'a {
        self.db
            .doc_id_prefix_to_indices(doc_id_prefix)
            .map(|r| r.map_err(|e| SearchEngineError::Other(e.into())))
    }

    pub fn search_with_bm25(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let voikko = VoikkoTokenizer::thread_local()?;

        assert!(self.main_record.doc_count >= self.main_record.deleted_docs);
        let n = (self.main_record.doc_count - self.main_record.deleted_docs) as f64;
        let k1 = self.main_record.options.bm25_k1;
        let b = self.main_record.options.bm25_b;
        let delta = self.main_record.options.bm25_delta;
        let avg_doc_words =
            self.main_record.total_doc_words as f64 / self.main_record.doc_count as f64;

        let query_tokens = voikko.tokenize(query);
        if query_tokens.len() == 0 {
            return Ok(Vec::new());
        }

        type ScoreMap = HashMap<u64, f64>;
        #[inline]
        fn add_to_score_map(scores: &mut ScoreMap, doc_index: u64, score: f64) {
            match scores.entry(doc_index) {
                Entry::Occupied(mut e) => {
                    *e.get_mut() += score;
                }
                Entry::Vacant(e) => {
                    e.insert(score);
                }
            }
        }
        fn combine_score_maps(mut a: ScoreMap, mut b: ScoreMap) -> ScoreMap {
            if a.len() < b.len() {
                // Do fewer insertions by inserting to the larger map
                mem::swap(&mut a, &mut b);
            }
            for (doc_index, score) in b {
                add_to_score_map(&mut a, doc_index, score);
            }
            a
        }

        let total_scores = query_tokens
            .par_iter()
            .map(|query_token| -> Result<ScoreMap> {
                query_token
                    .lemmas
                    .par_iter()
                    .try_fold(
                        || ScoreMap::new(),
                        |initial_scores, q| -> Result<ScoreMap> {
                            // https://en.wikipedia.org/wiki/Okapi_BM25
                            let mut lemma_entry_buf = Vec::new();
                            for entry in self.db.get_lemma_entries(&q) {
                                let entry = entry?;
                                lemma_entry_buf.push(entry);
                            }

                            let nq = lemma_entry_buf.len() as f64;
                            let idf = (1.0 + (n - nq + 0.5) / (nq + 0.5)).ln();

                            let scores = lemma_entry_buf
                                .par_iter()
                                .try_fold(
                                    || ScoreMap::new(),
                                    |mut scores, entry| -> Result<ScoreMap> {
                                        match self.db.get_doc_without_id(entry.doc_index)? {
                                            None => {}
                                            Some(doc) if doc.deleted => {}
                                            Some(doc) => {
                                                let freq = entry.freq as f64;
                                                let doc_words = doc.word_count as f64;

                                                let score = idf
                                                    * (delta
                                                        + freq * (k1 + 1.0)
                                                            / (freq
                                                                + k1 * (1.0 - b
                                                                    + b * doc_words
                                                                        / avg_doc_words)));
                                                add_to_score_map(
                                                    &mut scores,
                                                    entry.doc_index,
                                                    score,
                                                );
                                            }
                                        };
                                        Ok(scores)
                                    },
                                )
                                .try_reduce(
                                    || ScoreMap::new(),
                                    |a, b| Ok(combine_score_maps(a, b)),
                                )?;
                            Ok(combine_score_maps(initial_scores, scores))
                        },
                    )
                    .try_reduce(|| ScoreMap::new(), |a, b| Ok(combine_score_maps(a, b)))
            })
            .try_reduce(|| ScoreMap::new(), |a, b| Ok(combine_score_maps(a, b)))?;

        let mut pairs: Vec<(u64, f64)> = total_scores.into_iter().collect();
        pairs.par_sort_unstable_by_key(|&(i, s)| (FloatOrd(-s), i));
        let mut results: Vec<SearchResult> = Vec::with_capacity(pairs.len());
        for (index, score) in pairs {
            if results.len() >= limit {
                break;
            }
            if let Some(doc_id) = self.db.doc_index_to_id(index)? {
                results.push(SearchResult { doc_id, score });
            } else {
                return Err(SearchEngineError::Other(anyhow!(
                    "no ID for doc at index {}",
                    index
                )));
            }
        }
        Ok(results)
    }

    pub fn search_by_embedding(
        &self,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let embedding_len_floats = self.main_record.options.embedding_len as usize;
        let embedding_len_bytes = embedding_len_floats * mem::size_of::<f32>();

        let mut scores: Vec<(usize, f64)> = (0..(self.main_record.doc_count as usize))
            .into_par_iter()
            .map(|doc_index| {
                let byte_start = doc_index * embedding_len_bytes;
                let byte_end = byte_start + embedding_len_bytes;
                let embedding = unsafe {
                    slice::from_raw_parts(
                        self.embeddings_mmap[byte_start..byte_end].as_ptr() as *const f32,
                        embedding_len_floats,
                    )
                };
                let similarity = f32::cosine(query_embedding, embedding).unwrap();
                (doc_index, similarity)
            })
            .collect();

        scores.par_sort_by(|(_, s1), (_, s2)| s1.partial_cmp(&s2).unwrap_or(Ordering::Equal));

        let mut results: Vec<SearchResult> = Vec::with_capacity(limit.min(scores.len()));
        for &(doc_index, score) in scores.iter() {
            if results.len() >= limit {
                break;
            }
            if let Some(doc) = self.db.get_doc(doc_index as u64)? {
                if !doc.deleted {
                    results.push(SearchResult {
                        doc_id: doc.id,
                        score,
                    });
                }
            } else {
                return Err(SearchEngineError::Other(anyhow!(
                    "Doc at index {} not found",
                    doc_index
                )));
            }
        }
        Ok(results)
    }

    pub fn compact_into_dir(&self, dir: &Path) -> Result<SearchEngine> {
        let dir: PathBuf = dir.to_path_buf();
        let mut new_search = Self::new(&dir, self.main_record.options.clone())?;

        let mut batch = Vec::<Document>::new();
        for (i, doc) in read_json_lines_from_file_if_exists::<Document>(&self.paths.originals)?
            .into_iter()
            .enumerate()
        {
            match self.db.get_doc_without_id(i as u64)? {
                None => continue,
                Some(doc) if doc.deleted => continue,
                Some(_) => {}
            }

            batch.push(doc?);
            if batch.len() >= 1000 {
                new_search.add_batch(&batch)?;
                batch.clear();
            }
        }
        new_search.add_batch(&batch)?;

        Ok(new_search)
    }

    pub fn swap_with(&mut self, other: &mut SearchEngine) -> Result<()> {
        let self_dir = self
            .paths
            .main
            .parent()
            .ok_or(anyhow!("couldn't get parent dir of main path"))?;
        let other_dir = other
            .paths
            .main
            .parent()
            .ok_or(anyhow!("couldn't get parent dir of main path"))?;

        let self_dir_cstr: CString = CString::new(self_dir.as_os_str().as_bytes())
            .map_err(|_| anyhow!("can't convert main path to C string"))?;
        let other_dir_cstr: CString = CString::new(other_dir.as_os_str().as_bytes())
            .map_err(|_| anyhow!("can't convert main path to C string"))?;

        // Databases can't simply be swapped because they store the paths of files they use internally.
        // We mustn't return with an error before we've reopened the databases.
        self.db = DerefOption::None;
        other.db = DerefOption::None;

        unsafe {
            let result = libc::renameat2(
                libc::AT_FDCWD,
                self_dir_cstr.as_ptr() as *const i8,
                libc::AT_FDCWD,
                other_dir_cstr.as_ptr() as *const i8,
                libc::RENAME_EXCHANGE,
            );
            if result == -1 {
                panic!(
                    "renameat2 to exchange \"{}\" and \"{}\" failed: {:?}",
                    self_dir.display(),
                    other_dir.display(),
                    io::Error::last_os_error()
                );
            }
            assert!(result == 0);
        }

        std::mem::swap(self, other);
        std::mem::swap(&mut self.paths, &mut other.paths);

        // Databases need to be reopened.
        // We can't recover from this into a consistent state.
        // Designing a safe API to avoid this potential panic is possible, but not worth the effort.
        self.db = DerefOption::Some(
            SearchDB::open(&self.paths.db, self.main_record.last_committed_db_tx)
                .expect("failed to reopen swapped-in DB"),
        );
        other.db = DerefOption::Some(
            SearchDB::open(&other.paths.db, other.main_record.last_committed_db_tx)
                .expect("failed to reopen swapped-out DB"),
        );

        Ok(())
    }

    pub fn top_lemmas(&self, limit: usize) -> Result<Vec<(String, u64)>> {
        self.db.top_lemmas(limit).map_err(|e| e.into())
    }
}

impl Options {
    pub fn validate(&self) -> Result<()> {
        if self.embedding_len == 0 {
            return Err(SearchEngineError::BadSettingsError(
                "Embedding length cannot be zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResult {
    pub doc_id: String,
    pub score: f64,
}

const LOCK_FILE_NAME: &str = "lock";
const MAIN_FILE_NAME: &str = "main.json";
const ORIGINALS_FILE_NAME: &str = "originals.jsonl";
const EMBEDDINGS_FILE_NAME: &str = "embeddings.bin";
const DB_FILE_NAME: &str = "db";

#[derive(Serialize, Deserialize, Clone, Debug)]
struct MainRecord {
    machine_type: String,
    options: Options,
    doc_count: u64, // Includes deleted cocs
    deleted_docs: u64,
    total_doc_words: u64, // Does not include words of deleted docs
    originals_file_len: u64,
    embeddings_file_len: u64,
    last_committed_db_tx: u64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(rename = "emb")]
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub(crate) struct SummarizedDocument {
    pub(crate) index: u64,
    pub(crate) id: String,
    pub(crate) tokens: Vec<Token>,
}

impl SummarizedDocument {
    pub fn from_original_document(
        doc: &Document,
        index: u64,
        voikko: &VoikkoTokenizer,
    ) -> SummarizedDocument {
        SummarizedDocument {
            index,
            id: doc.id.clone(),
            tokens: voikko.tokenize(&doc.text),
        }
    }
}

struct PreparedAdd {
    updated_main_record: MainRecord,
    new_embeddings_mmap: Mmap,
    existing_doc_ids: Vec<String>,
}

struct PreparedDelete {
    new_main_record: MainRecord,
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::function_name;

    static SETTINGS: Options = Options {
        embedding_len: 3,
        ..Options::default()
    };

    macro_rules! test_dir {
        () => {
            crate::testing::clear_and_get_test_dir(function_name!())
        };
    }

    #[test]
    fn basic_init() {
        let dir = test_dir!();
        SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
    }

    #[test]
    fn open_twice() {
        let dir = test_dir!();
        SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine 1");
        SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine 2");
    }

    #[test]
    fn bm25_search() {
        let dir = test_dir!();
        let mut s =
            SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        s.add_batch(&[
            Document {
                id: "doc1".to_string(),
                text: "kissa ja koira".to_string(),
                embedding: vec![1.0, 2.0, 3.0],
            },
            Document {
                id: "doc2".to_string(),
                text: "koira vaan".to_string(),
                embedding: vec![-3.0, -2.0, -1.0],
            },
        ])
        .expect("failed to add document batch");

        fn test_searching(s: &SearchEngine) {
            let results = s.search_with_bm25("kissa", 3).unwrap();
            assert!(results.iter().any(|r| r.doc_id == "doc1"));
            assert!(!results.iter().any(|r| r.doc_id == "doc2"));

            let results = s.search_with_bm25("koira", 3).unwrap();
            assert!(results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));

            let results = s.search_with_bm25("vaan", 3).unwrap();
            assert!(!results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));
        }

        test_searching(&s);

        // Retest after reopening
        mem::drop(s); // Release lock
        s = SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        test_searching(&s);
    }

    #[test]
    fn embedding_search() {
        let dir = test_dir!();
        let mut s =
            SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        s.add_batch(&[
            Document {
                id: "doc1".to_string(),
                text: "kissa".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            Document {
                id: "doc2".to_string(),
                text: "kissanpentu".to_string(),
                embedding: vec![1.0, 0.5, 0.0],
            },
            Document {
                id: "doc3".to_string(),
                text: "koira".to_string(),
                embedding: vec![-1.0, 0.0, 0.0],
            },
        ])
        .expect("failed to add document batch");

        fn test_searching(s: &SearchEngine) {
            let results = s.search_by_embedding(&[1.0, 0.2, 0.0], 2).unwrap();
            assert_eq!(results[0].doc_id, "doc1");
            assert_eq!(results[1].doc_id, "doc2");

            let results = s.search_by_embedding(&[-1.0, 0.4, 0.0], 2).unwrap();
            assert_eq!(results[0].doc_id, "doc3");
            assert_eq!(results[1].doc_id, "doc2");
        }

        test_searching(&s);

        // Retest after reopening
        mem::drop(s); // Release lock
        s = SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        test_searching(&s);
    }

    #[test]
    fn cannot_add_same_doc_id() {
        let dir = test_dir!();
        let mut s =
            SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        assert!(s
            .add_batch(&[Document {
                id: "doc1".to_string(),
                text: "kissa".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
            }])
            .is_ok());
        let result = s
            .add_batch(&[
                Document {
                    id: "doc1".to_string(),
                    text: "koira".to_string(),
                    embedding: vec![0.0, 1.0, 0.0],
                },
                Document {
                    id: "doc2".to_string(),
                    text: "kissanpentu".to_string(),
                    embedding: vec![0.0, 0.0, 1.0],
                },
            ])
            .expect("add_batch failed");
        assert_eq!(result.existing_doc_ids, vec!["doc1".to_string()]);

        let results = s.search_by_embedding(&[0.0, 0.0, 1.0], 3).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "doc2");
        assert_eq!(results[1].doc_id, "doc1");
    }

    #[test]
    fn deleting() {
        let dir = test_dir!();
        let mut s =
            SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        s.add_batch(&[
            Document {
                id: "doc1".to_string(),
                text: "kissa ja koira".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            Document {
                id: "doc2".to_string(),
                text: "koira ja kissa".to_string(),
                embedding: vec![1.1, 0.0, 0.0],
            },
        ])
        .expect("failed to add document batch");

        let doc1_index = s.doc_id_to_index("doc1").unwrap().unwrap();
        s.delete(doc1_index).expect("failed to delete document");

        fn check_search_engine(s: &SearchEngine) {
            assert_eq!(s.main_record.total_doc_words, 3);
            assert_eq!(s.main_record.doc_count, 2);
            assert_eq!(s.main_record.deleted_docs, 1);

            let results = s.search_with_bm25("kissa", 3).unwrap();
            assert!(!results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));

            let results = s.search_by_embedding(&[1.0, 0.0, 0.0], 3).unwrap();
            assert!(!results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));
        }

        check_search_engine(&s);

        // Retest after reopening
        mem::drop(s); // Release lock
        s = SearchEngine::new(&dir, SETTINGS.clone()).expect("failed to create SearchEngine");
        check_search_engine(&s);
    }

    #[test]
    fn compaction() {
        let base_dir = test_dir!();
        let dir1 = base_dir.join("dir1");
        let dir2 = base_dir.join("dir2");
        let mut s =
            SearchEngine::new(&dir1, SETTINGS.clone()).expect("failed to create SearchEngine");

        s.add_batch(&[
            Document {
                id: "doc1".to_string(),
                text: "kissa ja koira".to_string(),
                embedding: vec![1.0, 0.0, 0.0],
            },
            Document {
                id: "doc2".to_string(),
                text: "koira ja kissa".to_string(),
                embedding: vec![1.1, 0.0, 0.0],
            },
        ])
        .expect("failed to add document batch");

        let doc1_index = s.doc_id_to_index("doc1").unwrap().unwrap();
        s.delete(doc1_index).expect("failed to delete document");

        s.compact_into_dir(&dir2)
            .expect("failed to compact into dir2");

        let mut s2 =
            SearchEngine::new(&dir2, SETTINGS.clone()).expect("failed to create SearchEngine");
        s.swap_with(&mut s2).expect("failed to swap with s2");

        fn check_search_engines(s: &SearchEngine, s2: &SearchEngine) {
            assert_eq!(s.main_record.doc_count, 1);
            assert_eq!(s2.main_record.doc_count, 2); // (Uncompacted)
            assert_eq!(s.main_record.total_doc_words, 3);

            let results = s.search_with_bm25("kissa", 3).unwrap();
            assert!(!results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));

            let results = s.search_by_embedding(&[1.0, 0.0, 0.0], 3).unwrap();
            assert!(!results.iter().any(|r| r.doc_id == "doc1"));
            assert!(results.iter().any(|r| r.doc_id == "doc2"));
        }

        check_search_engines(&s, &s2);

        // Retest after reopening
        mem::drop(s); // Release lock
        mem::drop(s2); // Release lock
        s = SearchEngine::new(&dir1, SETTINGS.clone()).expect("failed to create SearchEngine");
        s2 = SearchEngine::new(&dir2, SETTINGS.clone()).expect("failed to create SearchEngine");
        check_search_engines(&s, &s2);
    }
}
