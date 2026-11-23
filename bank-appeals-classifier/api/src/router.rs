use axum::{
    Router,
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    routing::post,
};

use crate::service::{PredictionError, PredictionService};

pub fn build(service: PredictionService) -> Router {
    Router::new()
        .route("/predict", post(predict))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .with_state(service)
}

async fn predict(
    State(service): State<PredictionService>,
    text: String,
) -> Result<String, (StatusCode, String)> {
    service.predict(text).await.map_err(|error| match error {
        PredictionError::InvalidInput(message) => (StatusCode::BAD_REQUEST, message),
        PredictionError::Internal => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Не удалось обработать обращение".into(),
        ),
    })
}
