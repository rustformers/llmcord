use anyhow::Context as AnyhowContext;
use rand::SeedableRng;
use serenity::{
    async_trait,
    builder::CreateComponents,
    client::{Context, EventHandler},
    futures::StreamExt,
    http::Http,
    model::{
        application::interaction::Interaction,
        prelude::{
            command::{Command, CommandOptionType},
            interaction::{
                application_command::ApplicationCommandInteraction, InteractionResponseType,
            },
            *,
        },
    },
    Client,
};
use std::{
    collections::HashSet,
    fmt::Display,
    sync::{Arc, Barrier},
};
use util::{run_and_report_error, DiscordInteraction};

mod config;
mod constant;
mod util;

use config::Configuration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    Configuration::init()?;

    let config = Configuration::get();
    let thread_count: i32 = config.inference.thread_count.try_into()?;

    let (request_tx, request_rx) = flume::unbounded::<GenerationRequest>();
    let (cancel_tx, cancel_rx) = flume::unbounded::<MessageId>();

    let barrier = Arc::new(Barrier::new(2));
    let model_thread = std::thread::spawn({
        let barrier = barrier.clone();

        fn process_token(
            token_tx: &flume::Sender<Token>,
            token: llama_rs::OutputToken,
        ) -> Result<(), SendError> {
            token_tx
                .send(match token {
                    llama_rs::OutputToken::Token(t) => Token::Token(t.to_string()),
                    llama_rs::OutputToken::EndOfText => unreachable!(),
                })
                .map_err(|_| SendError)
        }

        fn process_incoming_request(
            request: &GenerationRequest,
            model: &llama_rs::Model,
            vocab: &llama_rs::Vocabulary,
            cancel_rx: &flume::Receiver<MessageId>,
            thread_count: i32,
        ) -> anyhow::Result<()> {
            let mut rng = if let Some(seed) = request.seed {
                rand::rngs::StdRng::seed_from_u64(seed)
            } else {
                rand::rngs::StdRng::from_entropy()
            };

            let mut session = model.start_session(request.repeat_penalty_last_n_token_count);

            let params = llama_rs::InferenceParameters {
                n_threads: thread_count,
                n_batch: request.batch_size,
                top_k: request.top_k,
                top_p: request.top_p,
                repeat_penalty: request.repeat_penalty,
                temp: request.temperature,
            };

            session
                .feed_prompt(model, vocab, &params, &request.prompt, |t| {
                    process_token(&request.token_tx, t)
                })
                .map_err(|e| anyhow::Error::msg(e.to_string()))?;

            while let Ok(token) = session.infer_next_token(model, vocab, &params, &mut rng) {
                let cancellation_requests: HashSet<_> = cancel_rx.drain().collect();
                if cancellation_requests.contains(&request.message_id) {
                    request
                        .token_tx
                        .send(Token::Error("The generation was cancelled.".to_string()))
                        .map_err(|_| SendError)?;
                    break;
                }

                if token == llama_rs::OutputToken::EndOfText {
                    break;
                }

                process_token(&request.token_tx, token)?;
            }

            Ok(())
        }

        move || {
            let (model, vocab) = llama_rs::Model::load(
                &config.model.path,
                config.model.context_token_length.try_into().unwrap(),
                |_| {},
            )
            .unwrap();

            barrier.wait();

            loop {
                if let Ok(request) = request_rx.try_recv() {
                    match process_incoming_request(
                        &request,
                        &model,
                        &vocab,
                        &cancel_rx,
                        thread_count,
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
            }
        }
    });

    barrier.wait();

    let mut client = Client::builder(
        config
            .authentication
            .discord_token
            .as_deref()
            .context("Expected authentication.discord_token to be filled in config")?,
        GatewayIntents::default(),
    )
    .event_handler(Handler {
        _model_thread: model_thread,
        request_tx,
        cancel_tx,
    })
    .await
    .context("Error creating client")?;

    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }

    Ok(())
}

#[derive(Debug)]
struct SendError;
impl Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "failed to send message to channel")
    }
}
impl std::error::Error for SendError {}

struct GenerationRequest {
    prompt: String,
    batch_size: usize,
    repeat_penalty: f32,
    repeat_penalty_last_n_token_count: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    token_tx: flume::Sender<Token>,
    message_id: MessageId,
    seed: Option<u64>,
}

enum Token {
    Token(String),
    Error(String),
}

struct Handler {
    _model_thread: std::thread::JoinHandle<()>,
    request_tx: flume::Sender<GenerationRequest>,
    cancel_tx: flume::Sender<MessageId>,
}

async fn ready_handler(http: &Http) -> anyhow::Result<()> {
    let config = Configuration::get();

    let registered_commands = Command::get_global_application_commands(http).await?;
    let registered_commands: HashSet<_> = registered_commands
        .iter()
        .map(|c| c.name.as_str())
        .collect();

    let our_commands: HashSet<_> = config
        .commands
        .iter()
        .filter(|(_, v)| v.enabled)
        .map(|(k, _)| k.as_str())
        .collect();

    if registered_commands != our_commands {
        // If the commands registered with Discord don't match the commands configured
        // for this bot, reset them entirely.
        Command::set_global_application_commands(http, |c| c.set_application_commands(vec![]))
            .await?;
    }

    for (name, command) in config.commands.iter().filter(|(_, v)| v.enabled) {
        Command::create_global_application_command(http, |cmd| {
            cmd.name(name)
                .description(command.description.as_str())
                .create_option(|opt| {
                    opt.name(constant::value::PROMPT)
                        .description("The prompt.")
                        .kind(CommandOptionType::String)
                        .required(true)
                });

            create_parameters(cmd)
        })
        .await?;
    }

    Ok(())
}

fn create_parameters(
    command: &mut serenity::builder::CreateApplicationCommand,
) -> &mut serenity::builder::CreateApplicationCommand {
    command
        .create_option(|opt| {
            opt.name(constant::value::REPEAT_PENALTY)
                .kind(CommandOptionType::Number)
                .description("The penalty for repeating tokens. Higher values make the generation less likely to get into a loop.")
                .min_number_value(0.0)
                .required(false)
        })
        .create_option(|opt| {
            opt.name(constant::value::REPEAT_PENALTY_TOKEN_COUNT)
                .kind(CommandOptionType::Integer)
                .description("Size of the 'last N' buffer that is considered for the repeat penalty (in tokens)")
                .min_int_value(0)
                .max_int_value(64)
                .required(false)
        })
        .create_option(|opt| {
            opt.name(constant::value::TEMPERATURE)
                .kind(CommandOptionType::Number)
                .description("The temperature used for sampling.")
                .min_number_value(0.0)
                .required(false)
        })
        .create_option(|opt| {
            opt.name(constant::value::TOP_K)
                .kind(CommandOptionType::Integer)
                .description("The top K words by score are kept during sampling.")
                .min_int_value(0)
                .max_int_value(128)
                .required(false)
        })
        .create_option(|opt| {
            opt.name(constant::value::TOP_P)
                .kind(CommandOptionType::Number)
                .description("The cumulative probability after which no more words are kept for sampling.")
                .min_number_value(0.0)
                .max_number_value(1.0)
                .required(false)
        })
        .create_option(|opt| {
            opt.name(constant::value::SEED)
                .kind(CommandOptionType::Integer)
                .description("The seed to use for sampling.")
                .min_int_value(0)
                .required(false)
        })
}

#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected; registering commands...", ready.user.name);

        if let Err(err) = ready_handler(&ctx.http).await {
            println!("Error while registering commands: `{err}`");
            std::process::exit(1);
        }

        println!("{} is good to go!", ready.user.name);
    }

    async fn interaction_create(&self, ctx: Context, interaction: Interaction) {
        let http = &ctx.http;
        match interaction {
            Interaction::ApplicationCommand(cmd) => {
                let name = cmd.data.name.as_str();
                let commands = &Configuration::get().commands;

                if let Some(command) = commands.get(name) {
                    run_and_report_error(
                        &cmd,
                        http,
                        hallucinate(&cmd, http, self.request_tx.clone(), command),
                    )
                    .await;
                }
            }
            Interaction::MessageComponent(cmp) => {
                if cmp.data.custom_id == "cancel"
                    && cmp
                        .message
                        .interaction
                        .as_ref()
                        .map(|i| i.user == cmp.user)
                        .unwrap_or_default()
                {
                    self.cancel_tx.send(cmp.message.id).ok();
                    cmp.create_interaction_response(http, |r| {
                        r.kind(InteractionResponseType::DeferredUpdateMessage)
                    })
                    .await
                    .ok();
                }
            }
            _ => {}
        };
    }
}

async fn hallucinate(
    cmd: &ApplicationCommandInteraction,
    http: &Http,
    request_tx: flume::Sender<GenerationRequest>,
    command: &config::Command,
) -> anyhow::Result<()> {
    use constant::value as v;
    use util::{value_to_integer, value_to_number, value_to_string};

    let inference = &Configuration::get().inference;

    let options = &cmd.data.options;
    let user_prompt = util::get_value(options, v::PROMPT)
        .and_then(value_to_string)
        .context("no prompt specified")?;

    let user_prompt = if inference.replace_newlines {
        user_prompt.replace("\\n", "\n")
    } else {
        user_prompt
    };

    let processed_prompt = command.prompt.replace("{{PROMPT}}", &user_prompt);

    cmd.create_interaction_response(http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message
                    .content(format!(
                        "~~{}~~",
                        if inference.show_prompt_template {
                            &processed_prompt
                        } else {
                            &user_prompt
                        }
                    ))
                    .allowed_mentions(|m| m.empty_roles().empty_users().empty_parse())
            })
    })
    .await?;

    let message_id = cmd.get_interaction_message(http).await?.id;

    let repeat_penalty = util::get_value(options, v::REPEAT_PENALTY)
        .and_then(value_to_number)
        .unwrap_or(1.3) as f32;

    let repeat_penalty_last_n_token_count: usize =
        util::get_value(options, v::REPEAT_PENALTY_TOKEN_COUNT)
            .and_then(value_to_integer)
            .unwrap_or(64)
            .try_into()?;

    let temperature = util::get_value(options, v::TEMPERATURE)
        .and_then(value_to_number)
        .unwrap_or(0.8) as f32;

    let top_k: usize = util::get_value(options, v::TOP_K)
        .and_then(value_to_integer)
        .unwrap_or(40)
        .try_into()?;

    let top_p = util::get_value(options, v::TOP_P)
        .and_then(value_to_number)
        .unwrap_or(0.95) as f32;

    let seed = util::get_value(options, v::SEED)
        .and_then(value_to_integer)
        .map(|i| i as u64);

    let (token_tx, token_rx) = flume::unbounded();
    request_tx.send(GenerationRequest {
        prompt: processed_prompt.clone(),
        batch_size: inference.batch_size,
        repeat_penalty,
        repeat_penalty_last_n_token_count,
        temperature,
        top_k,
        top_p,
        token_tx,
        message_id,
        seed,
    })?;

    let mut seen_token = false;

    let last_update_duration =
        std::time::Duration::from_millis(inference.discord_message_update_interval_ms);

    let mut message = String::new();

    let mut stream = token_rx.into_stream();
    let mut last_update = std::time::Instant::now();

    while let Some(token) = stream.next().await {
        if !seen_token {
            if let Ok(mut r) = cmd.get_interaction_response(http).await {
                r.edit(http, |r| {
                    let mut components = CreateComponents::default();
                    components.create_action_row(|r| {
                        r.create_button(|b| {
                            b.custom_id("cancel")
                                .style(component::ButtonStyle::Danger)
                                .label("Cancel")
                        })
                    });
                    r.set_components(components)
                })
                .await?;
            }
        }

        match token {
            Token::Token(t) => {
                message += t.as_str();
                seen_token = true;
            }
            Token::Error(err) => {
                message = format!("Error: {err}");
                break;
            }
        }

        if last_update.elapsed() > last_update_duration {
            update_msg(
                cmd,
                http,
                &message,
                &user_prompt,
                &processed_prompt,
                &command.prompt,
                inference.show_prompt_template,
            )
            .await?;
            last_update = std::time::Instant::now();
        }
    }

    if let Ok(mut r) = cmd.get_interaction_response(http).await {
        r.edit(http, |r| r.set_components(CreateComponents::default()))
            .await?;
    }

    update_msg(
        cmd,
        http,
        &message,
        &user_prompt,
        &processed_prompt,
        &command.prompt,
        inference.show_prompt_template,
    )
    .await?;

    async fn update_msg(
        cmd: &ApplicationCommandInteraction,
        http: &Http,
        message: &str,
        user_prompt: &str,
        processed_prompt: &str,
        prompt_template: &str,
        show_prompt_template: bool,
    ) -> anyhow::Result<()> {
        let (message, display_prompt) = if !show_prompt_template {
            (
                fixup_message(message, user_prompt, prompt_template),
                user_prompt,
            )
        } else {
            (message.to_string(), processed_prompt)
        };

        let output = match message.strip_prefix(display_prompt) {
            Some(msg) => format!("**{display_prompt}**{msg}"),
            None => match display_prompt.strip_prefix(&message) {
                Some(ungenerated) => {
                    if message.is_empty() {
                        format!("~~{ungenerated}~~")
                    } else {
                        format!("**{message}**~~{ungenerated}~~")
                    }
                }
                None => message.to_string(),
            },
        };

        if !output.is_empty() {
            cmd.edit(http, &output).await?;
        }

        Ok(())
    }

    fn fixup_message(message: &str, prompt: &str, prompt_template: &str) -> String {
        if message.starts_with("Error: ") {
            return message.to_string();
        }

        let (prefix, suffix) = if prompt_template.contains("{{PROMPT}}") {
            let (prefix, suffix) = prompt_template.split_once("{{PROMPT}}").unwrap();
            (prefix, suffix)
        } else {
            ("", "")
        };

        let Some(message) = message.strip_prefix(prefix) else { return String::new(); };
        let Some(response) = message.strip_prefix(prompt) else { return message.to_string(); };
        let Some(response) = response.strip_prefix(suffix) else { return prompt.to_string(); };

        let newline = if suffix.ends_with('\n') { "\n" } else { "" };

        format!("{prompt}{newline}{response}")
    }

    Ok(())
}
