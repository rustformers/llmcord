use serenity::{
    async_trait,
    http::Http,
    model::{
        prelude::{
            interaction::{
                application_command::{
                    ApplicationCommandInteraction, CommandDataOption, CommandDataOptionValue,
                },
                message_component::MessageComponentInteraction,
                modal::ModalSubmitInteraction,
                InteractionResponseType,
            },
            ChannelId, GuildId, Message,
        },
        user::User,
    },
};
use std::future::Future;

pub fn get_value<'a>(
    options: &'a [CommandDataOption],
    name: &'a str,
) -> Option<&'a CommandDataOptionValue> {
    options
        .iter()
        .find(|v| v.name == name)
        .and_then(|v| v.resolved.as_ref())
}

pub fn value_to_string(v: &CommandDataOptionValue) -> Option<String> {
    match v {
        CommandDataOptionValue::String(v) => Some(v.clone()),
        _ => None,
    }
}

pub fn value_to_number(v: &CommandDataOptionValue) -> Option<f64> {
    match v {
        CommandDataOptionValue::Number(v) => Some(*v),
        _ => None,
    }
}

pub fn value_to_integer(v: &CommandDataOptionValue) -> Option<i64> {
    match v {
        CommandDataOptionValue::Integer(v) => Some(*v),
        _ => None,
    }
}

#[async_trait]
pub trait DiscordInteraction: Send + Sync {
    async fn create(&self, http: &Http, message: &str) -> anyhow::Result<()>;
    async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message>;
    async fn edit(&self, http: &Http, message: &str) -> anyhow::Result<()>;
    async fn create_or_edit(&self, http: &Http, message: &str) -> anyhow::Result<()>;

    fn channel_id(&self) -> ChannelId;
    fn guild_id(&self) -> Option<GuildId>;
    fn message(&self) -> Option<&Message>;
    fn user(&self) -> &User;
}
macro_rules! implement_interaction {
    ($name:ident) => {
        #[async_trait]
        impl DiscordInteraction for $name {
            async fn create(&self, http: &Http, msg: &str) -> anyhow::Result<()> {
                Ok(self
                    .create_interaction_response(http, |response| {
                        response
                            .kind(InteractionResponseType::ChannelMessageWithSource)
                            .interaction_response_data(|message| message.content(msg))
                    })
                    .await?)
            }
            async fn get_interaction_message(&self, http: &Http) -> anyhow::Result<Message> {
                Ok(self.get_interaction_response(http).await?)
            }
            async fn edit(&self, http: &Http, message: &str) -> anyhow::Result<()> {
                Ok(self
                    .get_interaction_message(http)
                    .await?
                    .edit(http, |m| m.content(message))
                    .await?)
            }
            async fn create_or_edit(&self, http: &Http, message: &str) -> anyhow::Result<()> {
                Ok(
                    if let Ok(mut msg) = self.get_interaction_message(http).await {
                        msg.edit(http, |m| m.content(message)).await?
                    } else {
                        self.create(http, message).await?
                    },
                )
            }

            fn channel_id(&self) -> ChannelId {
                self.channel_id
            }
            fn guild_id(&self) -> Option<GuildId> {
                self.guild_id
            }
            fn user(&self) -> &User {
                &self.user
            }
            interaction_message!($name);
        }
    };
}
macro_rules! interaction_message {
    (ApplicationCommandInteraction) => {
        fn message(&self) -> Option<&Message> {
            None
        }
    };
    (MessageComponentInteraction) => {
        fn message(&self) -> Option<&Message> {
            Some(&self.message)
        }
    };
    (ModalSubmitInteraction) => {
        fn message(&self) -> Option<&Message> {
            self.message.as_ref()
        }
    };
}
implement_interaction!(ApplicationCommandInteraction);
implement_interaction!(MessageComponentInteraction);
implement_interaction!(ModalSubmitInteraction);

/// Runs the [body] and edits the interaction response if an error occurs.
pub async fn run_and_report_error(
    interaction: &dyn DiscordInteraction,
    http: &Http,
    body: impl Future<Output = anyhow::Result<()>>,
) {
    if let Err(err) = body.await {
        interaction
            .create_or_edit(http, &format!("Error: {err}"))
            .await
            .unwrap();
    }
}
