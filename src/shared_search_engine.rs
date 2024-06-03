use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{self, AtomicBool},
};

use tokio::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::search_engine::SearchEngine;

pub struct SharedSearchEngine {
    state: RwLock<SharedState>,
}

struct SharedState {
    search_engine: SearchEngine,
    compaction_in_progress: AtomicBool,
}

impl SharedSearchEngine {
    pub fn new(search_engine: SearchEngine) -> Self {
        Self {
            state: RwLock::new(SharedState {
                search_engine,
                compaction_in_progress: AtomicBool::new(false),
            }),
        }
    }

    pub async fn read_lock(&self) -> ReadLock<'_> {
        ReadLock {
            guard: self.state.read().await,
        }
    }

    pub async fn write_lock(&self) -> Result<WriteLock<'_>, SharedSearchEngineError> {
        let guard = self.state.write().await;
        if guard.compaction_in_progress.load(atomic::Ordering::SeqCst) {
            return Err(SharedSearchEngineError::CompactionOngoing);
        }
        Ok(WriteLock { guard })
    }

    pub async fn write_lock_allow_ongoing_compaction(&self) -> WriteLock<'_> {
        let guard = self.state.write().await;
        WriteLock { guard }
    }
}

#[derive(Debug)]
pub enum SharedSearchEngineError {
    CompactionOngoing,
}

pub struct ReadLock<'a> {
    guard: RwLockReadGuard<'a, SharedState>,
}

impl<'a> Deref for ReadLock<'a> {
    type Target = SearchEngine;
    fn deref(&self) -> &Self::Target {
        &self.guard.deref().search_engine
    }
}

pub struct WriteLock<'a> {
    guard: RwLockWriteGuard<'a, SharedState>,
}

impl<'a> Deref for WriteLock<'a> {
    type Target = SearchEngine;
    fn deref(&self) -> &Self::Target {
        &self.guard.deref().search_engine
    }
}

impl<'a> DerefMut for WriteLock<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard.deref_mut().search_engine
    }
}

impl<'a> WriteLock<'a> {
    pub fn is_compaction_in_progress(&self) -> bool {
        self.guard
            .compaction_in_progress
            .load(atomic::Ordering::SeqCst)
    }

    pub fn mark_compaction_complete(&self) {
        self.guard
            .compaction_in_progress
            .store(false, atomic::Ordering::SeqCst);
    }

    pub fn into_compaction_lock(self) -> Result<CompactionLock<'a>, SharedSearchEngineError> {
        let prev = self
            .guard
            .compaction_in_progress
            .swap(true, atomic::Ordering::SeqCst);
        if prev == true {
            return Err(SharedSearchEngineError::CompactionOngoing);
        }
        Ok(CompactionLock {
            guard: self.guard.downgrade(),
        })
    }
}

pub struct CompactionLock<'a> {
    guard: RwLockReadGuard<'a, SharedState>,
}

impl<'a> Deref for CompactionLock<'a> {
    type Target = SearchEngine;
    fn deref(&self) -> &Self::Target {
        &self.guard.search_engine
    }
}
