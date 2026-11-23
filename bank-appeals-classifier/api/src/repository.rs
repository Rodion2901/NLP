use std::{path::Path, time::Duration};

use rusqlite::{Connection, params};

pub struct PredictionRepository(Connection);

impl PredictionRepository {
    pub fn open(path: &Path) -> rusqlite::Result<Self> {
        let connection = Connection::open(path)?;
        connection.busy_timeout(Duration::from_secs(5))?;
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                request_text TEXT NOT NULL,
                response_text TEXT NOT NULL
            );",
        )?;
        Ok(Self(connection))
    }

    pub fn save(&self, request: &str, response: &str) -> rusqlite::Result<()> {
        self.0.execute(
            "INSERT INTO predictions (request_text, response_text) VALUES (?1, ?2)",
            params![request, response],
        )?;
        Ok(())
    }
}
