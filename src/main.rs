use axum::{debug_handler, extract::State, http::StatusCode, routing::post, Json, Router};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;
mod llama_cpp;
use llama_cpp::LlamaApp;
mod prompts;
use prompts::*;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_line_number(true)
        .with_thread_names(true)
        .init();

    let model_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("Usage: {} <model_path>", std::env::args().next().unwrap());
        std::process::exit(1);
    });
    info!("Starting server with model file: {}", model_path);
    let llama = LlamaApp::new(&model_path).expect("Failed to initialize LLaMA backend");
    let model = Arc::new(llama);
    info!("Model initialized successfully");
    let app = Router::new()
        .route("/classify_ticket", post(classify_ticket))
        .with_state(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:3000")
        .await
        .unwrap();
    info!("Listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
}

fn get_prompt(data: String) -> String {
    CLASSIFY_TICKET_PROMPT.replace("{}", &data)
}

#[debug_handler]
async fn classify_ticket(
    State(model): State<Arc<LlamaApp>>,
    Json(request): Json<ClassifyTicketRequest>,
) -> (StatusCode, Json<ClassifyTicketResponse>) {
    info!("Received request data: {:?}", request.text);
    let prompt = get_prompt(request.text);
    let agent_response = model.generate_text(&prompt, 512, 1.0, Some(42)).unwrap();
    let agent_response: ClassifyTicketResponse = serde_json::from_str(&agent_response).unwrap();
    info!("Agent response: {:?}", agent_response);
    (StatusCode::OK, Json(agent_response))
}

// the input to our `classify_ticket` handler
#[derive(Deserialize)]
struct ClassifyTicketRequest {
    text: String,
}

// the output to our `classify_ticket` handler
#[derive(Serialize, Debug, Clone, Deserialize)]
struct ClassifyTicketResponse {
    product_name: String,
    brief_description: String,
    issue_classification: String,
}

