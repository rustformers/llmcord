use anyhow::Context;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct Configuration {
    pub authentication: Authentication,
    pub model: Model,
    pub inference: Inference,
    pub commands: HashMap<String, Command>,
}
impl Default for Configuration {
    fn default() -> Self {
        Self {
            authentication: Authentication {
                discord_token: None,
            },
            model: Model {
                path: "models/7B/ggml-alpaca-q4_0.bin".to_string(),
                context_token_length: 2048,
            },
            inference: Inference {
                thread_count: 8,
                discord_message_update_interval_ms: 250,
                replace_newlines: true,
                show_prompt_template: true,
            },
            commands: HashMap::from_iter([
                (
                    "hallucinate".into(),
                    Command {
                        enabled: false,
                        description: "Hallucinates some text.".into(),
                        prompt: "{PROMPT}".into(),
                    },
                ),
                (
                    "alpaca".into(),
                    Command {
                        enabled: false,
                        description: "Responds to the provided instruction.".into(),
                        prompt: indoc::indoc! {
                            "Below is an instruction that describes a task. Write a response that appropriately completes the request.

                            ### Instruction:
                            
                            {{PROMPT}}
                            
                            ### Response:
                            
                            "
                        }.into(),
                    },
                ),
            ]),
        }
    }
}
impl Configuration {
    const FILENAME: &str = "config.toml";

    pub fn init() -> anyhow::Result<()> {
        CONFIGURATION
            .set(Self::load()?)
            .ok()
            .context("config already set")
    }

    pub fn get() -> &'static Self {
        CONFIGURATION.wait()
    }

    fn load() -> anyhow::Result<Self> {
        let config = if let Ok(file) = std::fs::read_to_string(Self::FILENAME) {
            toml::from_str(&file).context("failed to load config")?
        } else {
            let config = Self::default();
            config.save()?;
            config
        };

        Ok(config)
    }

    fn save(&self) -> anyhow::Result<()> {
        Ok(std::fs::write(
            Self::FILENAME,
            toml::to_string_pretty(self)?,
        )?)
    }
}
static CONFIGURATION: OnceCell<Configuration> = OnceCell::new();

#[derive(Serialize, Deserialize, Debug)]
pub struct Authentication {
    pub discord_token: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub path: String,
    pub context_token_length: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Inference {
    pub thread_count: usize,
    /// Low values will result in you getting throttled by Discord
    pub discord_message_update_interval_ms: u64,
    /// Whether or not to replace '\n' with newlines
    pub replace_newlines: bool,
    /// Whether or not to show the entire prompt template, or just
    /// what the user specified
    pub show_prompt_template: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Command {
    pub enabled: bool,
    pub description: String,
    pub prompt: String,
}
