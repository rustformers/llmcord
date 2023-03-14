use anyhow::Context;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Configuration {
    pub authentication: Authentication,
    pub model: Model,
    pub inference: Inference,
    pub commands: Commands,
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
impl Default for Authentication {
    fn default() -> Self {
        Self {
            discord_token: None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Model {
    pub path: String,
    pub context_token_length: usize,
}
impl Default for Model {
    fn default() -> Self {
        Self {
            path: "models/7B/ggml-model-q4_0.bin".to_string(),
            context_token_length: 512,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Inference {
    pub thread_count: usize,
    /// Low values will result in you getting throttled by Discord
    pub discord_message_update_interval_ms: u64,
    /// Whether or not to replace '\n' with newlines
    pub replace_newlines: bool,
}
impl Default for Inference {
    fn default() -> Self {
        Self {
            thread_count: 8,
            discord_message_update_interval_ms: 250,
            replace_newlines: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Commands {
    pub hallucinate: String,
}
impl Commands {
    pub fn all(&self) -> HashSet<&str> {
        HashSet::from_iter([self.hallucinate.as_str()])
    }
}
impl Default for Commands {
    fn default() -> Self {
        Self {
            hallucinate: "hallucinate".to_string(),
        }
    }
}
