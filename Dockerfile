# Use the official Rust image as the builder
FROM rust:1.70 AS build

# create a new empty shell project
RUN USER=root cargo new --bin llmcord
WORKDIR /llmcord

# copy over your manifests
COPY ./Cargo.lock ./Cargo.lock
COPY ./Cargo.toml ./Cargo.toml

# this build step will cache your dependencies
RUN cargo build --release
RUN rm src/*.rs

# copy your source tree
COPY ./config.toml ./config.toml
COPY ./src ./src

# build for release
RUN rm ./target/release/deps/llmcord*
RUN cargo build --release

########### Start final stage ###########

# Use the official Rust image as the final base image
FROM rust:1.70

WORKDIR /usr/src/llmcord

# copy the build artifact from the build stage
COPY --from=build /llmcord/config.toml .
COPY --from=build /llmcord/target/release/llmcord .

# Run the application
CMD ["./llmcord"]
