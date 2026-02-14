use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use uuid::Uuid;

const DOCUMENTS_TABLE: TableDefinition<u128, &str> = TableDefinition::new("documents");
const EMBEDDINGS_TABLE: TableDefinition<u128, &[u8]> = TableDefinition::new("embeddings");

#[derive(Debug)]
pub struct VectorStore {
    db: Database,
}

impl VectorStore {
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let db = Database::create(path)?;
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(DOCUMENTS_TABLE)?;
            let _ = write_txn.open_table(EMBEDDINGS_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    pub fn add(&self, text: String, vector: Vec<f32>) -> Result<Uuid> {
        if text.trim().is_empty() {
            bail!("text must not be empty");
        }
        validate_vector(&vector, "vector")?;

        let doc_id = Uuid::new_v4();
        let key = doc_id.as_u128();
        let embedding =
            bincode::serialize(&vector).context("failed to serialize embedding vector")?;

        let write_txn = self.db.begin_write()?;
        {
            let mut documents = write_txn.open_table(DOCUMENTS_TABLE)?;
            documents.insert(key, text.as_str())?;

            let mut embeddings = write_txn.open_table(EMBEDDINGS_TABLE)?;
            embeddings.insert(key, embedding.as_slice())?;
        }
        write_txn.commit()?;
        Ok(doc_id)
    }

    pub fn search(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        if top_k == 0 {
            bail!("top_k must be greater than 0");
        }
        validate_vector(query_vec, "query_vec")?;

        let query_norm = l2_norm(query_vec);
        if query_norm <= f32::EPSILON {
            bail!("query_vec norm must be greater than zero");
        }

        let read_txn = self.db.begin_read()?;
        let documents = read_txn.open_table(DOCUMENTS_TABLE)?;
        let embeddings = read_txn.open_table(EMBEDDINGS_TABLE)?;
        let mut heap: BinaryHeap<Reverse<ScoredDocument>> = BinaryHeap::new();

        for entry in embeddings.iter()? {
            let (id_guard, vector_guard) = entry?;
            let doc_id = id_guard.value();
            let bytes = vector_guard.value();
            let vector: Vec<f32> = bincode::deserialize(bytes)
                .with_context(|| format!("failed to deserialize embedding for doc {doc_id}"))?;

            if vector.len() != query_vec.len() {
                continue;
            }
            if vector.iter().any(|v| !v.is_finite()) {
                continue;
            }

            let Some(score) = cosine_similarity(query_vec, &vector, query_norm) else {
                continue;
            };

            let candidate = ScoredDocument { id: doc_id, score };
            if heap.len() < top_k {
                heap.push(Reverse(candidate));
                continue;
            }

            if let Some(current_min) = heap.peek() {
                if candidate.score > current_min.0.score {
                    heap.pop();
                    heap.push(Reverse(candidate));
                }
            }
        }

        let mut ranked: Vec<ScoredDocument> = heap.into_iter().map(|item| item.0).collect();
        ranked.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });

        let mut results = Vec::with_capacity(ranked.len());
        for item in ranked {
            let Some(text_guard) = documents.get(item.id)? else {
                continue;
            };
            results.push((text_guard.value().to_owned(), item.score));
        }
        Ok(results)
    }
}

#[derive(Debug, Clone, Copy)]
struct ScoredDocument {
    id: u128,
    score: f32,
}

impl PartialEq for ScoredDocument {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for ScoredDocument {}

impl PartialOrd for ScoredDocument {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDocument {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

fn validate_vector(vector: &[f32], field: &str) -> Result<()> {
    if vector.is_empty() {
        return Err(anyhow!("{field} must not be empty"));
    }
    if vector.iter().any(|v| !v.is_finite()) {
        return Err(anyhow!("{field} contains non-finite values"));
    }
    Ok(())
}

fn l2_norm(vector: &[f32]) -> f32 {
    vector.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn cosine_similarity(query_vec: &[f32], candidate: &[f32], query_norm: f32) -> Option<f32> {
    let mut dot = 0.0_f32;
    let mut candidate_norm_sq = 0.0_f32;
    for (&left, &right) in query_vec.iter().zip(candidate.iter()) {
        dot += left * right;
        candidate_norm_sq += right * right;
    }

    let candidate_norm = candidate_norm_sq.sqrt();
    if candidate_norm <= f32::EPSILON {
        return None;
    }
    Some(dot / (query_norm * candidate_norm))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::VectorStore;

    #[test]
    fn add_and_search_returns_ranked_documents() {
        let temp = tempdir().expect("temp dir");
        let db_path = temp.path().join("memory.redb");
        let store = VectorStore::new(&db_path).expect("create store");

        store
            .add("sum function".to_string(), vec![1.0, 0.0, 0.0])
            .expect("insert first");
        store
            .add("auth service".to_string(), vec![0.0, 1.0, 0.0])
            .expect("insert second");
        store
            .add("state machine".to_string(), vec![0.7, 0.2, 0.0])
            .expect("insert third");

        let results = store.search(&[1.0, 0.0, 0.0], 2).expect("search");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "sum function");
        assert!(results[0].1 >= results[1].1);
    }

    #[test]
    fn search_validates_arguments() {
        let temp = tempdir().expect("temp dir");
        let db_path = temp.path().join("memory.redb");
        let store = VectorStore::new(&db_path).expect("create store");

        let err = store
            .search(&[], 1)
            .expect_err("empty query vector must fail");
        assert!(err.to_string().contains("query_vec"));

        let err = store.search(&[1.0, 2.0], 0).expect_err("top_k=0 must fail");
        assert!(err.to_string().contains("top_k"));
    }

    #[test]
    fn search_skips_dimension_mismatch() {
        let temp = tempdir().expect("temp dir");
        let db_path = temp.path().join("memory.redb");
        let store = VectorStore::new(&db_path).expect("create store");

        store
            .add("two dims".to_string(), vec![0.4, 0.9])
            .expect("insert");
        let results = store.search(&[1.0, 0.0, 0.0], 5).expect("search");
        assert!(results.is_empty());
    }
}
