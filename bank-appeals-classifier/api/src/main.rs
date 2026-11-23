mod repository;
mod router;
mod service;

use std::{env, error::Error, path::Path};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("API directory must have a parent");
    let python = env::var_os("BANK_API_PYTHON").unwrap_or_else(|| {
        let local = root.join("venv/bin/python");
        if local.is_file() {
            local.into_os_string()
        } else {
            "python3".into()
        }
    });
    let db = env::var_os("BANK_API_DB")
        .map(Into::into)
        .unwrap_or_else(|| root.join("api/predictions.sqlite3"));
    let address = env::var("BANK_API_ADDR").unwrap_or_else(|_| "127.0.0.1:3000".into());

    let service = service::PredictionService::new(root, &python, &db)?;
    let listener = tokio::net::TcpListener::bind(&address).await?;
    eprintln!("Listening on {}", listener.local_addr()?);
    axum::serve(listener, router::build(service)).await?;
    Ok(())
}
