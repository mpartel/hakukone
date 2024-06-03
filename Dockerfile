
# *** Build container ***

FROM rust:1-slim-bookworm AS builder

RUN apt-get update \
 && apt-get install -y llvm-dev libclang-dev clang libsnappy-dev libvoikko-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Build dependencies in separate layer for faster recompiles on source-only changes
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src \
 && echo 'fn main() { println!("Placeholder!"); }' > src/main.rs \
 && cargo build --release \
 && rm -f src/main.rs

COPY README.md LICENSE.txt ./

COPY src/ ./src/
RUN touch src/main.rs && cargo build --release


# *** Final container ***

FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y voikko-fi libvoikko1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/target/release/hakukone /build/README.md /build/LICENSE.txt ./

ENTRYPOINT ["./hakukone"]
EXPOSE 3888
