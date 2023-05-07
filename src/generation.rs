use std::{collections::HashSet, fmt::Display, thread::JoinHandle};

use rand::SeedableRng;
use serenity::model::prelude::MessageId;

use crate::config::Configuration;

#[derive(Debug)]
pub struct CustomInferenceError(String);
impl CustomInferenceError {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}
impl Display for CustomInferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for CustomInferenceError {}

pub struct Request {
    pub prompt: String,
    pub batch_size: usize,
    pub repeat_penalty: f32,
    pub repeat_penalty_last_n_token_count: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub token_tx: flume::Sender<Token>,
    pub message_id: MessageId,
    pub seed: Option<u64>,
}

pub enum Token {
    Token(String),
    Error(String),
}

pub fn make_thread(
    model: Box<dyn llm::Model>,
    config: Configuration,
    request_rx: flume::Receiver<Request>,
    cancel_rx: flume::Receiver<MessageId>,
) -> JoinHandle<()> {
    std::thread::spawn(move || loop {
        if let Ok(request) = request_rx.try_recv() {
            match process_incoming_request(
                &request,
                model.as_ref(),
                &cancel_rx,
                config.inference.thread_count,
            ) {
                Ok(_) => {}
                Err(e) => {
                    if let Err(err) = request.token_tx.send(Token::Error(e.to_string())) {
                        eprintln!("Failed to send error: {err:?}");
                    }
                }
            }
        }

        std::thread::sleep(std::time::Duration::from_millis(5));
    })
}

fn process_incoming_request(
    request: &Request,
    model: &dyn llm::Model,
    cancel_rx: &flume::Receiver<MessageId>,
    thread_count: usize,
) -> anyhow::Result<()> {
    let mut rng = if let Some(seed) = request.seed {
        rand::rngs::StdRng::seed_from_u64(seed)
    } else {
        rand::rngs::StdRng::from_entropy()
    };

    let mut session = model.start_session(llm::InferenceSessionParameters {
        repetition_penalty_last_n: request.repeat_penalty_last_n_token_count,
        ..Default::default()
    });

    let params = llm::InferenceParameters {
        n_threads: thread_count,
        n_batch: request.batch_size,
        top_k: request.top_k,
        top_p: request.top_p,
        repeat_penalty: request.repeat_penalty,
        temperature: request.temperature,
        bias_tokens: Default::default(),
    };

    session
        .infer_with_params(
            model,
            &params,
            &Default::default(),
            &request.prompt,
            &mut Default::default(),
            &mut rng,
            move |t| {
                let cancellation_requests: HashSet<_> = cancel_rx.drain().collect();
                if cancellation_requests.contains(&request.message_id) {
                    return Err(CustomInferenceError::new("The generation was cancelled."));
                }

                request
                    .token_tx
                    .send(Token::Token(t.to_string()))
                    .map_err(|_| CustomInferenceError::new("Failed to send token to channel."))?;

                Ok(())
            },
        )
        .map_err(|e| {
            anyhow::Error::msg(match e {
                llm::InferenceError::UserCallback(e) => e.to_string(),
                e => e.to_string(),
            })
        })?;

    Ok(())
}
