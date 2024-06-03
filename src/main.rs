use std::{io, mem, path::PathBuf, sync::Arc, time::Instant};

use actix_web::{error, middleware, post, web, App, HttpServer, Result};
use clap::{arg, Parser, Subcommand};
use hakukone::{
    import::import_jsonl,
    search_engine::{
        AddBatchResult, Document, Options, SearchEngine, SearchEngineError, SearchResult,
    },
    shared_search_engine::{CompactionLock, SharedSearchEngine, SharedSearchEngineError},
    voikko_tokenizer::VoikkoTokenizer,
};
use log::info;
use serde::{Deserialize, Serialize};
use tempfile::tempdir_in;

#[derive(Serialize, Deserialize)]
struct SearchByEmbeddingRequest {
    embedding: Vec<f32>,
    limit: usize,
}

#[derive(Serialize, Deserialize)]
struct SearchByTextRequest {
    query: String,
    limit: usize,
}

#[derive(Serialize, Deserialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
}

#[derive(Serialize, Deserialize)]
struct AddBatchRequest {
    docs: Vec<Document>,
}

#[derive(Serialize, Deserialize)]
struct AddBatchResponse {
    existing_doc_ids: Vec<String>,
}

#[derive(Serialize, Deserialize)]
struct DeleteRequest {
    doc_id: String,
}

#[derive(Serialize, Deserialize)]
struct DeleteByPrefixRequest {
    doc_id_prefix: String,
}

#[post("/api/search/by-embedding")]
async fn search_by_embedding(
    req: web::Json<SearchByEmbeddingRequest>,
    app_data: web::Data<AppData>,
) -> Result<web::Json<SearchResponse>> {
    let search_engine = app_data.search_engine.read_lock().await;
    match search_engine.search_by_embedding(&req.embedding, req.limit) {
        Ok(results) => Ok(web::Json(SearchResponse { results })),
        Err(e) => {
            return Err(error::ErrorInternalServerError(format!(
                "Error searching with embedding: {:?}",
                e
            )));
        }
    }
}

#[post("/api/search/by-text")]
async fn search_by_text(
    req: web::Json<SearchByTextRequest>,
    app_data: web::Data<AppData>,
) -> Result<web::Json<SearchResponse>> {
    let search_engine = app_data.search_engine.read_lock().await;
    match search_engine.search_with_bm25(&req.query, req.limit) {
        Ok(results) => Ok(web::Json(SearchResponse { results })),
        Err(e) => {
            return Err(error::ErrorInternalServerError(format!(
                "Error searching with query: {:?}",
                e
            )));
        }
    }
}

#[post("/api/add-batch")]
async fn add_batch(
    req: web::Json<AddBatchRequest>,
    app_data: web::Data<AppData>,
) -> Result<web::Json<AddBatchResponse>> {
    let mut search_engine = match app_data.search_engine.write_lock().await {
        Ok(search_engine) => search_engine,
        Err(SharedSearchEngineError::CompactionOngoing) => {
            return Err(compaction_in_progress_error())
        }
    };
    match search_engine.add_batch(&req.docs) {
        Ok(AddBatchResult { existing_doc_ids }) => {
            Ok(web::Json(AddBatchResponse { existing_doc_ids }))
        }
        Err(e) => Err(error::ErrorInternalServerError(format!(
            "Error adding batch: {:?}",
            e
        ))),
    }
}

fn compaction_in_progress_error() -> error::Error {
    error::ErrorServiceUnavailable("Compacting in progress. Try again later.")
}

#[post("/api/delete")]
async fn delete(
    req: web::Json<DeleteRequest>,
    app_data: web::Data<AppData>,
) -> Result<web::Json<()>> {
    let mut search_engine = match app_data.search_engine.write_lock().await {
        Ok(search_engine) => search_engine,
        Err(SharedSearchEngineError::CompactionOngoing) => {
            return Err(compaction_in_progress_error())
        }
    };
    if let Some(doc_index) = search_engine
        .doc_id_to_index(&req.doc_id)
        .map_err(|e| error::ErrorInternalServerError(format!("Error getting document: {:?}", e)))?
    {
        if let Err(e) = search_engine.delete(doc_index) {
            return Err(error::ErrorInternalServerError(format!(
                "Error deleting document: {:?}",
                e
            )));
        }
    }
    Ok(web::Json(()))
}

#[post("/api/delete/by-prefix")]
async fn delete_by_prefix(
    req: web::Json<DeleteByPrefixRequest>,
    app_data: web::Data<AppData>,
) -> Result<web::Json<()>> {
    let mut search_engine = match app_data.search_engine.write_lock().await {
        Ok(search_engine) => search_engine,
        Err(SharedSearchEngineError::CompactionOngoing) => {
            return Err(compaction_in_progress_error())
        }
    };
    let indices: Vec<Result<u64, SearchEngineError>> = search_engine
        .doc_id_prefix_to_indices(&req.doc_id_prefix)
        .collect();
    for index in indices {
        let index = index.map_err(|e| error::ErrorInternalServerError(e))?;
        println!("DEL {}", index);
        search_engine.delete(index).map_err(|e| {
            error::ErrorInternalServerError(format!("Error deleting document: {:?}", e))
        })?;
    }
    Ok(web::Json(()))
}

#[post("/api/compact")]
async fn compact(app_data: web::Data<AppData>) -> Result<web::Json<()>> {
    compact_or_clear(app_data, false)
        .await
        .map(|_| web::Json(()))
}

#[post("/api/clear-database")]
async fn clear_database(app_data: web::Data<AppData>) -> Result<web::Json<()>> {
    compact_or_clear(app_data, true)
        .await
        .map(|_| web::Json(()))
}

async fn compact_or_clear(app_data: web::Data<AppData>, clear: bool) -> Result<()> {
    let compaction_lock: CompactionLock = match app_data
        .search_engine
        .write_lock_allow_ongoing_compaction()
        .await
        .into_compaction_lock()
    {
        Ok(search_engine) => search_engine,
        Err(SharedSearchEngineError::CompactionOngoing) => {
            return Err(compaction_in_progress_error())
        }
    };

    let tempdir = tempdir_in(&app_data.data_dir)?;
    if !clear {
        if let Err(e) = compaction_lock.compact_into_dir(tempdir.path()) {
            return Err(error::ErrorInternalServerError(format!(
                "Error compacting: {:?}",
                e
            )));
        };
    }

    mem::drop(compaction_lock);
    let mut write_lock = app_data
        .search_engine
        .write_lock_allow_ongoing_compaction()
        .await;
    let mut compacted = SearchEngine::new(tempdir.path(), (*app_data.settings).clone())
        .map_err(|e| error::ErrorInternalServerError(e))?;
    write_lock
        .swap_with(&mut compacted)
        .map_err(|e| error::ErrorInternalServerError(e))?;
    write_lock.mark_compaction_complete();
    Ok(())
}

struct AppData {
    data_dir: PathBuf,
    search_engine: Arc<SharedSearchEngine>,
    settings: Arc<Options>,
}

#[derive(Parser, Debug)]
struct CliArgs {
    #[command(subcommand)]
    command: Command,
}

const DEFAULT_DATA_DIR: &str = "./data";

#[derive(Subcommand, Debug)]
enum Command {
    Serve {
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        #[arg(long, default_value_t = 3888)]
        port: u16,

        #[arg(long, default_value = DEFAULT_DATA_DIR)]
        data_dir: String,

        #[arg(long)]
        embedding_len: u64,
    },

    ImportJsonl {
        #[arg(long, default_value = DEFAULT_DATA_DIR)]
        data_dir: String,

        #[arg(long)]
        embedding_len: u64,
    },

    TopLemmas {
        #[arg(long, default_value = DEFAULT_DATA_DIR)]
        data_dir: String,

        #[arg(long, default_value_t = 100)]
        limit: usize,

        #[arg(long)]
        embedding_len: u64,
    },
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli_args = CliArgs::parse();
    match cli_args.command {
        Command::Serve {
            host,
            port,
            data_dir,
            embedding_len,
        } => {
            // Test initializing Voikko, which needs external libraries and data files.
            // Otherwise we'd only catch errors when a text search request comes in.
            VoikkoTokenizer::new()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

            let data_dir = PathBuf::from(data_dir);
            let live_data_dir = data_dir.join("live");

            let settings = Arc::new({
                let mut settings = Options::default();
                settings.embedding_len = embedding_len;
                settings
            });
            let search_engine: Arc<SharedSearchEngine> = {
                info!("Opening database...");
                let t0 = Instant::now();
                let search_engine = Arc::new(SharedSearchEngine::new(
                    SearchEngine::new(&live_data_dir, (*settings).clone())
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?,
                ));
                info!(
                    "Opening database took {:.2}s",
                    (Instant::now() - t0).as_secs_f64()
                );
                search_engine
            };
            let server = HttpServer::new(move || {
                App::new()
                    .configure(configure_app(
                        data_dir.clone(),
                        search_engine.clone(),
                        settings.clone(),
                    ))
                    .wrap(middleware::Logger::default())
            })
            .bind((host, port))?;
            info!("Starting server at port {}", port);
            server.run().await?;
            Ok(())
        }

        Command::ImportJsonl {
            data_dir,
            embedding_len,
        } => {
            let data_dir = PathBuf::from(data_dir);
            let live_data_dir = data_dir.join("live");
            let mut settings = Options::default();
            settings.embedding_len = embedding_len;
            let mut search_engine = SearchEngine::new(&live_data_dir, settings)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            import_jsonl(io::stdin(), &mut search_engine)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            Ok(())
        }

        Command::TopLemmas {
            data_dir,
            limit,
            embedding_len,
        } => {
            let data_dir = PathBuf::from(data_dir);
            let live_data_dir = data_dir.join("live");
            let mut settings = Options::default();
            settings.embedding_len = embedding_len;
            let search_engine = SearchEngine::new(&live_data_dir, settings)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            let top_lemmas = search_engine
                .top_lemmas(limit)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            for (lemma, doc_count) in top_lemmas {
                println!("{} {}", doc_count, lemma);
            }
            Ok(())
        }
    }
}

fn configure_app(
    data_dir: PathBuf,
    search_engine: Arc<SharedSearchEngine>,
    settings: Arc<Options>,
) -> impl for<'a> FnOnce(&'a mut web::ServiceConfig) {
    move |cfg: &mut web::ServiceConfig| {
        cfg.app_data(web::Data::new(AppData {
            data_dir,
            search_engine,
            settings,
        }))
        .service(search_by_embedding)
        .service(search_by_text)
        .service(add_batch)
        .service(delete)
        .service(delete_by_prefix)
        .service(compact)
        .service(clear_database);
    }
}

#[cfg(test)]
mod test {
    use std::path::Path;

    use super::*;
    use actix_web::{dev, test};
    use hakukone::function_name;

    macro_rules! test_dir {
        () => {
            hakukone::testing::clear_and_get_test_dir(function_name!())
        };
    }

    static SETTINGS: Options = Options {
        embedding_len: 3,
        ..Options::default()
    };

    fn test_app_config(data_dir: &Path) -> impl FnOnce(&mut web::ServiceConfig) {
        let search_engine = SearchEngine::new(&data_dir.join("live"), SETTINGS.clone()).unwrap();
        let shared_search_engine = Arc::new(SharedSearchEngine::new(search_engine));
        configure_app(
            data_dir.to_path_buf(),
            shared_search_engine,
            Arc::new(SETTINGS.clone()),
        )
    }

    async fn check_response(resp: dev::ServiceResponse) -> dev::ServiceResponse {
        if !resp.status().is_success() {
            panic!(
                "Response failed: {:?}: {:?}",
                resp.status(),
                test::read_body(resp.map_into_boxed_body()).await
            );
        }
        resp
    }

    #[actix_web::test]
    async fn basic_happy_path() {
        let dir = test_dir!();
        let app = test::init_service(App::new().configure(test_app_config(&dir))).await;

        let req = test::TestRequest::post()
            .uri("/api/add-batch")
            .set_json(AddBatchRequest {
                docs: vec![
                    Document {
                        id: "doc1".to_string(),
                        text: "kissa & kissa".to_string(),
                        embedding: vec![1.0, 0.0, 0.0],
                    },
                    Document {
                        id: "doc2".to_string(),
                        text: "kissa ja koira".to_string(),
                        embedding: vec![1.0, 1.0, 0.0],
                    },
                ],
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/search/by-embedding")
            .set_json(SearchByEmbeddingRequest {
                embedding: vec![1.0, 0.0, 0.0],
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].doc_id, "doc1");
        assert_eq!(resp.results[1].doc_id, "doc2");

        let req = test::TestRequest::post()
            .uri("/api/search/by-text")
            .set_json(SearchByTextRequest {
                query: "kissa".to_string(),
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].doc_id, "doc1");
        assert_eq!(resp.results[1].doc_id, "doc2");

        let req = test::TestRequest::post()
            .uri("/api/search/by-text")
            .set_json(SearchByTextRequest {
                query: "koira".to_string(),
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].doc_id, "doc2");

        let req = test::TestRequest::post()
            .uri("/api/delete")
            .set_json(DeleteRequest {
                doc_id: "doc1".to_string(),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/search/by-embedding")
            .set_json(SearchByEmbeddingRequest {
                embedding: vec![1.0, 0.0, 0.0],
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].doc_id, "doc2");
    }

    #[actix_web::test]
    async fn delete_by_prefix() {
        let dir = test_dir!();
        let app = test::init_service(App::new().configure(test_app_config(&dir))).await;

        let req = test::TestRequest::post()
            .uri("/api/add-batch")
            .set_json(AddBatchRequest {
                docs: vec![
                    Document {
                        id: "group1-doc1".to_string(),
                        text: "kissa 1".to_string(),
                        embedding: vec![1.0, 0.0, 0.0],
                    },
                    Document {
                        id: "group2-doc1".to_string(),
                        text: "kissa 2".to_string(),
                        embedding: vec![0.0, 1.0, 0.0],
                    },
                    Document {
                        id: "group1-doc2".to_string(),
                        text: "kissa 3".to_string(),
                        embedding: vec![2.0, 0.0, 0.0],
                    },
                ],
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/delete/by-prefix")
            .set_json(DeleteByPrefixRequest {
                doc_id_prefix: "group1-".to_string(),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/search/by-text")
            .set_json(SearchByTextRequest {
                query: "kissa".to_string(),
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].doc_id, "group2-doc1");
    }

    #[actix_web::test]
    async fn compaction() {
        let dir = test_dir!();
        let app = test::init_service(App::new().configure(test_app_config(&dir))).await;

        let req = test::TestRequest::post()
            .uri("/api/add-batch")
            .set_json(AddBatchRequest {
                docs: vec![
                    Document {
                        id: "doc1".to_string(),
                        text: "kissa & kissa".to_string(),
                        embedding: vec![1.0, 0.0, 0.0],
                    },
                    Document {
                        id: "doc2".to_string(),
                        text: "kissa ja koira".to_string(),
                        embedding: vec![1.0, 1.0, 0.0],
                    },
                ],
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/delete")
            .set_json(DeleteRequest {
                doc_id: "doc1".to_string(),
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post().uri("/api/compact").to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post().uri("/api/compact").to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/search/by-text")
            .set_json(SearchByTextRequest {
                query: "koira".to_string(),
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].doc_id, "doc2");
    }

    #[actix_web::test]
    async fn clearing() {
        let dir = test_dir!();
        let app = test::init_service(App::new().configure(test_app_config(&dir))).await;

        let req = test::TestRequest::post()
            .uri("/api/add-batch")
            .set_json(AddBatchRequest {
                docs: vec![
                    Document {
                        id: "doc1".to_string(),
                        text: "kissa & kissa".to_string(),
                        embedding: vec![1.0, 0.0, 0.0],
                    },
                    Document {
                        id: "doc2".to_string(),
                        text: "kissa ja koira".to_string(),
                        embedding: vec![1.0, 1.0, 0.0],
                    },
                ],
            })
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/clear-database")
            .to_request();
        let resp = test::call_service(&app, req).await;
        check_response(resp).await;

        let req = test::TestRequest::post()
            .uri("/api/search/by-text")
            .set_json(SearchByTextRequest {
                query: "koira".to_string(),
                limit: 3,
            })
            .to_request();
        let resp: SearchResponse = test::call_and_read_body_json(&app, req).await;
        assert_eq!(resp.results.len(), 0);
    }
}
