use anyhow::Context as AnyhowContext;
use serenity::{
    async_trait,
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
    let barrier = Arc::new(Barrier::new(2));
    let model_thread = std::thread::spawn({
        let barrier = barrier.clone();
        move || {
            let (model, vocab) = llama_rs::Model::load(
                &config.model.path,
                config.model.context_token_length.try_into().unwrap(),
                |_| {},
            )
            .unwrap();

            barrier.wait();

            let mut rng = rand::thread_rng();
            loop {
                if let Ok(request) = request_rx.try_recv() {
                    let token_tx = request.token_tx;
                    model.inference_with_prompt(
                        &vocab,
                        &llama_rs::InferenceParameters {
                            n_threads: thread_count,
                            n_predict: request.maximum_token_count,
                            n_batch: request.batch_size,
                            top_k: request.top_k.try_into().unwrap(),
                            top_p: request.top_p,
                            repeat_last_n: request.repeat_penalty_last_n_token_count,
                            repeat_penalty: request.repeat_penalty,
                            temp: request.temperature,
                        },
                        &request.prompt,
                        &mut rng,
                        {
                            let token_tx = token_tx.clone();
                            move |t| {
                                token_tx
                                    .send(match t {
                                        llama_rs::OutputToken::Token(t) => {
                                            Token::Token(t.to_string())
                                        }
                                        llama_rs::OutputToken::EndOfText => Token::EndOfText,
                                    })
                                    .unwrap();
                            }
                        },
                    );
                };

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
    })
    .await
    .context("Error creating client")?;

    if let Err(why) = client.start().await {
        println!("Client error: {why:?}");
    }

    Ok(())
}

struct GenerationRequest {
    prompt: String,
    maximum_token_count: usize,
    batch_size: usize,
    repeat_penalty: f32,
    repeat_penalty_last_n_token_count: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    token_tx: flume::Sender<Token>,
}

enum Token {
    Token(String),
    EndOfText,
}

struct Handler {
    _model_thread: std::thread::JoinHandle<()>,
    request_tx: flume::Sender<GenerationRequest>,
}

async fn ready_handler(http: &Http) -> anyhow::Result<()> {
    let config = Configuration::get();

    let registered_commands = Command::get_global_application_commands(http).await?;
    let registered_commands: HashSet<_> = registered_commands
        .iter()
        .map(|c| c.name.as_str())
        .collect();

    let our_commands: HashSet<_> = config.commands.all().iter().cloned().collect();

    if registered_commands != our_commands {
        // If the commands registered with Discord don't match the commands configured
        // for this bot, reset them entirely.
        Command::set_global_application_commands(http, |c| c.set_application_commands(vec![]))
            .await?;
    }

    Command::create_global_application_command(http, |command| {
        command
            .name(&config.commands.hallucinate)
            .description("Hallucinates some text using the LLaMA language model.")
            .create_option(|opt| {
                opt.name(constant::value::PROMPT)
                    .description("The prompt for LLaMA. Note that LLaMA requires autocomplete-like prompts.")
                    .kind(CommandOptionType::String)
                    .required(true)
            })
            .create_option(|opt| {
                opt.name(constant::value::MAXIMUM_TOKEN_COUNT)
                    .description("The maximum number of tokens to predict.")
                    .kind(CommandOptionType::Integer)
                    .min_int_value(0)
                    .max_int_value(512)
                    .required(false)
            })
            .create_option(|opt| {
                opt.name(constant::value::BATCH_SIZE)
                    .kind(CommandOptionType::Integer)
                    .description("The number of tokens taken from the prompt to feed the network. Does not affect generation.")
                    .min_int_value(0)
                    .max_int_value(64)
                    .required(false)
            })
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
                    .description("The cummulative probability after which no more words are kept for sampling.")
                    .min_number_value(0.0)
                    .max_number_value(1.0)
                    .required(false)
            })
    })
    .await?;

    Ok(())
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

                if name == commands.hallucinate {
                    run_and_report_error(
                        &cmd,
                        &http,
                        hallucinate(&cmd, &http, self.request_tx.clone()),
                    )
                    .await;
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
) -> anyhow::Result<()> {
    use constant::value as v;
    use util::{value_to_integer, value_to_number, value_to_string};

    let inference = &Configuration::get().inference;

    let options = &cmd.data.options;
    let prompt = util::get_value(options, v::PROMPT)
        .and_then(value_to_string)
        .context("no prompt specified")?;

    let prompt = if inference.replace_newlines {
        prompt.replace("\\n", "\n")
    } else {
        prompt
    };

    cmd.create_interaction_response(http, |response| {
        response
            .kind(InteractionResponseType::ChannelMessageWithSource)
            .interaction_response_data(|message| {
                message
                    .content(format!("~~{prompt}~~"))
                    .allowed_mentions(|m| m.empty_roles().empty_users().empty_parse())
            })
    })
    .await?;

    let maximum_token_count: usize = util::get_value(options, v::MAXIMUM_TOKEN_COUNT)
        .and_then(value_to_integer)
        .unwrap_or(128)
        .try_into()?;

    let batch_size: usize = util::get_value(options, v::BATCH_SIZE)
        .and_then(value_to_integer)
        .unwrap_or(8)
        .try_into()?;

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

    let (token_tx, token_rx) = flume::unbounded();
    request_tx.send(GenerationRequest {
        prompt: prompt.clone(),
        maximum_token_count,
        batch_size,
        repeat_penalty,
        repeat_penalty_last_n_token_count,
        temperature,
        top_k,
        top_p,
        token_tx,
    })?;

    let last_update_duration =
        std::time::Duration::from_millis(inference.discord_message_update_interval_ms);

    let mut message = String::new();
    let mut ended = false;

    let mut stream = token_rx.into_stream();
    let mut last_update = std::time::Instant::now();

    async fn update_msg(
        cmd: &ApplicationCommandInteraction,
        http: &Http,
        message: &str,
        prompt: &str,
    ) -> anyhow::Result<()> {
        let output = match message.strip_prefix(prompt) {
            Some(msg) => format!("**{prompt}**{msg}"),
            None => match prompt.strip_prefix(message) {
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

    while let Some(token) = stream.next().await {
        match token {
            Token::Token(t) => {
                message += t.as_str();
            }
            Token::EndOfText => {
                ended = true;
            }
        }

        if last_update.elapsed() > last_update_duration {
            update_msg(cmd, http, &message, &prompt).await?;
            last_update = std::time::Instant::now();
        }
    }
    if !ended {
        message += " [generation ended before message end]";
    }

    update_msg(cmd, http, &message, &prompt).await?;

    Ok(())
}
