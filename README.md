# llamacord

A Discord bot, written in Rust, that generates responses using the LLaMA language model.

Built on top of [llama-rs](https://github.com/setzer22/llama-rs).

## Setup

### Model

- Obtain the LLaMA weights from a reputable source (like Meta).
- Convert and quantize them to GGML-q4 format using [llama.cpp](https://github.com/ggerganov/llama.cpp#usage).

### Bot

#### Discord

- [Create a Discord application](https://discord.com/developers/applications) and fill it out with your own details.
- Go to `Bot` and create a new Bot.
  - Hit `Reset Token`, and copy the token it gives you somewhere.
- Go to `OAuth2 > URL Generator`, select `bot`, then select `Send Messages` and `Use Slash Commands`.
  - Go to the URL it generates, and then invite it to a server of your choice.

#### Application

- Install Rust 1.68 or above using `rustup`.
- Set the `RUSTFLAGS='-C target-feature=+avx2,+fma,+f16c'` environment variable before building for maximal optimisation.
- Run `cargo run --release` to start llamacord. This will auto-generate a configuration file, and then quit.
- Fill in the configuration file with the required details, including the path to the quantized model.
- You can then run llamacord to your heart's content.
