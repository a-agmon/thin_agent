# Thin Agent: Building LLM-Powered Microservices for Commodity Hardware

## Introduction

This repository is the companion code for the blog post ["Thin Agents: Creating Lean AI Services with Local Fine-Tuned LLMs"](https://towardsdatascience.com/thin-agents-creating-lean-ai-services-with-local-fine-tuned-llms-6253233d9798). It demonstrates how to create a simple, lean, and task-specific AI-powered microservice using Rust and Unsloth.

The project focuses on building a lightweight AI-powered ticket routing service for a call center application. This service uses a locally hosted, fine-tuned, and quantized version of Llama 3.2 1B to classify customer support tickets based on their content, improving routing accuracy without requiring extensive AI infrastructure.

For the full explanation of the implementation and methodology, refer to the blog post.

## Features

- Shows how to use Unsloth to quickly fine-tune an LLM.
- Use llama.cpp binding for Rust in order to run LLM inference on CPU.
- Create an Axum Rust service that wraps load and inference of fine-tuned models.

## How to Run

This is a Rust Axum service. To run it, use the following command:

```bash
cargo run --release -- <model gguf file path>
```

### Example Request
You can call the service with the following example:

```bash
curl -X POST http://127.0.0.1:3000/classify_ticket \
  -H 'Content-Type: application/json' \
  -d '{"text": "I just got the kettle and its not working"}'
```

