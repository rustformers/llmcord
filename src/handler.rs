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

        let _model_thread = generation::make_thread(model, request_rx, cancel_rx);
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
                if let ["cancel", message_id, user_id] =
                    cmp.data.custom_id.split('#').collect::<Vec<_>>()[..]
                {
                    if let (Ok(message_id), Ok(user_id)) =
                        (message_id.parse::<u64>(), user_id.parse::<u64>())
                    {
                        if cmp.user.id == user_id {
                            self.cancel_tx.send(MessageId(message_id)).ok();
                            cmp.create_interaction_response(http, |r| {
                                r.kind(InteractionResponseType::DeferredUpdateMessage)
                            })
                            .await
                            .ok();
                        }
                    }
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

    let mut outputter = Outputter::new(
        http,
        cmd,
        Prompts {
            show_prompt_template: inference.show_prompt_template,
            processed: command.prompt.replace("{{PROMPT}}", &user_prompt),
            user: user_prompt,
            template: command.prompt.clone(),
        },
        std::time::Duration::from_millis(inference.discord_message_update_interval_ms),
    )
    .await?;

    let message = cmd.get_interaction_message(http).await?;
    let message_id = message.id;

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
        prompt: outputter.prompts.processed.clone(),
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

    let mut stream = token_rx.into_stream();

    let mut errored = false;
    while let Some(token) = stream.next().await {
        match token {
            Token::Token(t) => {
                outputter.new_token(&t).await?;
            }
            Token::Error(err) => {
                match err {
                    generation::InferenceError::Cancelled => outputter.cancelled().await?,
                    generation::InferenceError::Custom(m) => outputter.error(&m).await?,
                };
                errored = true;
                break;
            }
        }
    }
    if !errored {
        outputter.finish().await?;
    }

    Ok(())
}

struct Prompts {
    show_prompt_template: bool,

    processed: String,
    user: String,
    template: String,
}
impl Prompts {
    fn make_markdown_message(&self, message: &str) -> String {
        let (message, display_prompt) = if !self.show_prompt_template {
            (self.decouple_prompt_from_message(message), &self.user)
        } else {
            (message.to_string(), &self.processed)
        };

        match message.strip_prefix(display_prompt) {
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
        }
    }

    fn decouple_prompt_from_message(&self, output: &str) -> String {
        let (prefix, suffix) = self.template.split_once("{{PROMPT}}").unwrap_or_default();

        let prompt = &self.user;

        let Some(message) = output.strip_prefix(prefix) else { return String::new(); };
        let Some(response) = message.strip_prefix(prompt) else { return message.to_string(); };
        let Some(response) = response.strip_prefix(suffix) else { return prompt.to_string(); };

        let newline = if suffix.ends_with('\n') { "\n" } else { "" };

        format!("{prompt}{newline}{response}")
    }
}

struct Outputter<'a> {
    http: &'a Http,

    user_id: UserId,
    messages: Vec<Message>,
    chunks: Vec<String>,

    message: String,
    prompts: Prompts,

    in_terminal_state: bool,

    last_update: std::time::Instant,
    last_update_duration: std::time::Duration,
}
impl<'a> Outputter<'a> {
    const MESSAGE_CHUNK_SIZE: usize = 1500;

    async fn new(
        http: &'a Http,
        cmd: &ApplicationCommandInteraction,
        prompts: Prompts,
        last_update_duration: std::time::Duration,
    ) -> anyhow::Result<Outputter<'a>> {
        cmd.create_interaction_response(http, |response| {
            response
                .kind(InteractionResponseType::ChannelMessageWithSource)
                .interaction_response_data(|message| {
                    message
                        .content(format!(
                            "~~{}~~",
                            if prompts.show_prompt_template {
                                &prompts.processed
                            } else {
                                &prompts.user
                            }
                        ))
                        .allowed_mentions(|m| m.empty_roles().empty_users().empty_parse())
                })
        })
        .await?;
        let starting_message = cmd.get_interaction_response(http).await?;

        Ok(Self {
            http,

            user_id: cmd.user.id,
            messages: vec![starting_message],
            chunks: vec![],

            message: String::new(),
            prompts,

            in_terminal_state: false,

            last_update: std::time::Instant::now(),
            last_update_duration,
        })
    }

    async fn new_token(&mut self, token: &str) -> anyhow::Result<()> {
        if self.in_terminal_state {
            return Ok(());
        }

        if self.message.is_empty() {
            // Add the cancellation button when we receive the first token
            if let Some(first) = self.messages.first_mut() {
                add_cancel_button(self.http, first.id, first, self.user_id).await?;
            }
        }

        self.message += token;

        // This could be much more efficient but that's a problem for later
        self.chunks = {
            let mut chunks: Vec<String> = vec![];

            let markdown = self.prompts.make_markdown_message(&self.message);
            for word in markdown.split(' ') {
                if let Some(last) = chunks.last_mut() {
                    if last.len() > Self::MESSAGE_CHUNK_SIZE {
                        chunks.push(word.to_string());
                    } else {
                        last.push(' ');
                        last.push_str(word);
                    }
                } else {
                    chunks.push(word.to_string());
                }
            }

            chunks
        };

        if self.last_update.elapsed() > self.last_update_duration {
            self.sync_messages_with_chunks().await?;
            self.last_update = std::time::Instant::now();
        }

        Ok(())
    }

    async fn error(&mut self, err: &str) -> anyhow::Result<()> {
        self.on_error(err).await
    }

    async fn cancelled(&mut self) -> anyhow::Result<()> {
        self.on_error("The generation was cancelled.").await
    }

    async fn finish(&mut self) -> anyhow::Result<()> {
        for msg in &mut self.messages {
            msg.edit(self.http, |m| m.set_components(CreateComponents::default()))
                .await?;
        }

        self.sync_messages_with_chunks().await?;

        Ok(())
    }

    async fn sync_messages_with_chunks(&mut self) -> anyhow::Result<()> {
        // Update the last message with its latest state, then insert the remaining chunks in one go
        if let Some((msg, chunk)) = self.messages.iter_mut().zip(self.chunks.iter()).last() {
            msg.edit(self.http, |m| m.content(chunk)).await?;
        }

        if self.chunks.len() <= self.messages.len() {
            return Ok(());
        }

        // Remove the cancel button from all existing messages
        for msg in &mut self.messages {
            msg.edit(self.http, |m| m.set_components(CreateComponents::default()))
                .await?;
        }

        // Create new messages for the remaining chunks
        let Some(first_id) = self.messages.first().map(|m| m.id) else { return Ok(()); };
        for chunk in self.chunks[self.messages.len()..].iter() {
            let last = self.messages.last_mut().unwrap();
            let msg = last.reply(self.http, chunk).await?;
            self.messages.push(msg);
        }

        // Add the cancel button to the last message
        if let Some(last) = self.messages.last_mut() {
            add_cancel_button(self.http, first_id, last, self.user_id).await?;
        }

        Ok(())
    }

    async fn on_error(&mut self, error_message: &str) -> anyhow::Result<()> {
        for msg in &mut self.messages {
            let cut_content = format!("~~{}~~", msg.content);
            msg.edit(self.http, |m| {
                m.set_components(CreateComponents::default())
                    .content(cut_content)
            })
            .await?;
        }

        let Some(last) = self.messages.last_mut() else { return Ok(()); };
        last.reply(self.http, error_message).await?;

        self.in_terminal_state = true;

        Ok(())
    }
}

async fn add_cancel_button(
    http: &Http,
    first_id: MessageId,
    msg: &mut Message,
    user_id: UserId,
) -> anyhow::Result<()> {
    Ok(msg
        .edit(http, |r| {
            let mut components = CreateComponents::default();
            components.create_action_row(|r| {
                r.create_button(|b| {
                    b.custom_id(format!("cancel#{first_id}#{user_id}"))
                        .style(component::ButtonStyle::Danger)
                        .label("Cancel")
                })
            });
            r.set_components(components)
        })
        .await?)
}
