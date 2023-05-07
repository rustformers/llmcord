# llmcord

![llmcord logo: a vaguely Discord Clyde-looking llama](docs/llmcord.png)

A Discord bot, written in Rust, that generates responses using any language model supported by `llm`.

Built on top of [llm](https://crates.io/crates/llm).

## Setup

### Model

See [llm's README](https://github.com/rustformers/llm#getting-models).

### Bot

#### Discord

- [Create a Discord application](https://discord.com/developers/applications) and fill it out with your own details.
- Go to `Bot` and create a new Bot.
  - Hit `Reset Token`, and copy the token it gives you somewhere.
- Go to `OAuth2 > URL Generator`, select `bot`, then select `Send Messages` and `Use Slash Commands`.
  - Go to the URL it generates, and then invite it to a server of your choice.

#### Application

- Install Rust 1.68 or above using `rustup`.
- Run `cargo run --release` to start llmcord. This will auto-generate a configuration file, and then quit.
- Fill in the configuration file with the required details, including the path to the model.
- You can then run llmcord to your heart's content.
