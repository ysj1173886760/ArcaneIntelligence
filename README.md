# ArcaneIntelligence

toy project. it's used for learning AutoGPT, LangChain and other things related to LLMs.

My goal is to build a customized AI assistant that can help me understand codebase, organize informations such as article or books, and increase the efficiency of my daily work.

# Architecture

I want to use the [architecture](https://github.com/Significant-Gravitas/AutoGPT/issues/4770) from AutoGPT.

Basically, we have Agent, Planning, Resources, Abilities and Memory. I've removed Plugins and Workspace since i don't know how to utilize them right now.

TODO(sheep): more architecture details

# Roadmap

[x] Implement LLM provider, chat with LLM first
[] Test LLAMA Index
  [] Implement llama index based on MoonshotAI
[] Test LangChain
[] Implement Planning
[] Implement Abilities
[] Combine above components and implement simple agent
[] Introduce memory subsystem