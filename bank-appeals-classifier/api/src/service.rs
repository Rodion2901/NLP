use std::{
    ffi::OsStr,
    io::{self, BufRead, BufReader, Write},
    path::Path,
    process::{Child, ChildStdin, ChildStdout, Command, Stdio},
    sync::Arc,
};

use serde::Deserialize;
use tokio::sync::Mutex;

use crate::repository::PredictionRepository;

#[derive(Clone)]
pub struct PredictionService(Arc<Mutex<ServiceState>>);

struct ServiceState {
    model: ModelWorker,
    repository: PredictionRepository,
}

pub enum PredictionError {
    InvalidInput(String),
    Internal,
}

impl PredictionService {
    pub fn new(root: &Path, python: &OsStr, db: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let model = ModelWorker::start(root, python)?;
        let repository = PredictionRepository::open(db)?;
        Ok(Self(Arc::new(Mutex::new(ServiceState {
            model,
            repository,
        }))))
    }

    pub async fn predict(&self, text: String) -> Result<String, PredictionError> {
        if text.trim().is_empty() {
            return Err(PredictionError::InvalidInput(
                "Передайте непустой текст".into(),
            ));
        }

        let mut state = self.0.clone().lock_owned().await;
        tokio::task::spawn_blocking(move || {
            let response = match state.model.predict(&text) {
                Ok(ModelReply::Label(label)) => label,
                Ok(ModelReply::Failed) => {
                    eprintln!("Python model prediction failed");
                    return Err(PredictionError::Internal);
                }
                Ok(ModelReply::Invalid(message)) => {
                    return Err(PredictionError::InvalidInput(message));
                }
                Err(error) => {
                    eprintln!("Model worker failed: {}", error.kind());
                    return Err(PredictionError::Internal);
                }
            };
            state.repository.save(&text, &response).map_err(|error| {
                eprintln!("SQLite insert failed: {error}");
                PredictionError::Internal
            })?;
            Ok(response)
        })
        .await
        .map_err(|error| {
            eprintln!("Prediction task failed: {error}");
            PredictionError::Internal
        })?
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelReply {
    Label(String),
    Invalid(String),
    Failed,
}

struct ModelWorker {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl ModelWorker {
    fn start(root: &Path, python: &OsStr) -> io::Result<Self> {
        let mut child = Command::new(python)
            .arg(root.join("api/model_worker.py"))
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;
        let input = child.stdin.take().expect("piped stdin");
        let output = BufReader::new(child.stdout.take().expect("piped stdout"));
        let mut worker = Self {
            child,
            input,
            output,
        };
        let mut ready = String::new();
        worker.output.read_line(&mut ready)?;
        if ready.trim_end() != "ready" {
            return Err(io::Error::other("Python model worker did not start"));
        }
        Ok(worker)
    }

    fn predict(&mut self, text: &str) -> io::Result<ModelReply> {
        serde_json::to_writer(&mut self.input, text)?;
        self.input.write_all(b"\n")?;
        self.input.flush()?;
        let mut line = String::new();
        self.output.read_line(&mut line)?;
        Ok(serde_json::from_str(&line)?)
    }
}

impl Drop for ModelWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
