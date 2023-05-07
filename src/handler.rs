use crate::{
    config::{self, Configuration},
    constant,
    generation::{self, Token},
    util::{self, run_and_report_error, DiscordInteraction},
};
use anyhow::Context as AnyhowContext;
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
};
use std::collections::HashSet;

pub struct Handler {
    _model_thread: std::thread::JoinHandle<()>,
    config: Configuration,
    request_tx: flume::Sender<generation::Request>,
    cancel_tx: flume::Sender<MessageId>,
}
impl Handler {
    pub fn new(config: Configuration, model: Box<dyn llm::Model>) -> Self {
        let (request_tx, request_rx) = flume::unbounded::<generation::Request>();
        let (cancel_tx, cancel_rx) = flume::unbounded::<MessageId>();

        let _model_thread = generation::make_thread(model, config.clone(), request_rx, cancel_rx);
        Self {
            _model_thread,
            config,
            request_tx,
            cancel_tx,
        }
    }
}
#[async_trait]
impl EventHandler for Handler {
    async fn ready(&self, ctx: Context, ready: Ready) {
        println!("{} is connected; registering commands...", ready.user.name);

        if let Err(err) = ready_handler(&ctx.http, &self.config).await {
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
                let commands = &self.config.commands;

                if let Some(command) = commands.get(name) {
                    run_and_report_error(
                        &cmd,
                        http,
                        hallucinate(
                            &cmd,
                            http,
                            self.request_tx.clone(),
                            &self.config.inference,
                            command,
                        ),
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

async fn ready_handler(http: &Http, config: &Configuration) -> anyhow::Result<()> {
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

async fn hallucinate(
    cmd: &ApplicationCommandInteraction,
    http: &Http,
    request_tx: flume::Sender<generation::Request>,
    inference: &config::Inference,
    command: &config::Command,
) -> anyhow::Result<()> {
    use constant::value as v;
    use util::{value_to_integer, value_to_number, value_to_string};

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
    request_tx.send(generation::Request {
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

    Ok(())
}

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
