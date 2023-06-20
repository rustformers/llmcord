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

Note that you can define your own commands in the configuration, like so:

```toml
[commands.makecaption]
enabled = true
description = "Attempts to make an image description for the given prompt."
prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:

Create an evocative image description for "{{PROMPT}}".

### Response:

"""
```

#### Docker
Before build you need to have the `config.toml` and the `/path/models` where the `model` you want to use

To build the Docker run the following command:
```sh
docker build -t llmcord:latest .
```
To run the Docker container and mount the `/path/models` directory on the host machine to the `/usr/src/llmcord/weights` directory inside the container, execute the following command:
```sh
docker run -v /path/models:/usr/src/llmcord/weights llmcord:latest
```
Ensure that you replace `/path/models` with the actual path where your model is on your host machine to mount the volume.

#### Docker Compose
set the `path` of the model you want to use in `docker-compose.yml`
```
docker compose -p llmcord up -d
```