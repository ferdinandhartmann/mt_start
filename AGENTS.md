# Project Agents.md Guide for OpenAI Codex

This Agents.md file provides comprehensive guidance for OpenAI Codex and other AI agents working with this codebase.

## Project Structure for OpenAI Codex Navigation

mobile_robot_rl_2_sac.ipynb is the main file I am working on. I am trying to implement rl with free energy principle and active inference on a simulated mobile robot.

- `/oldruns`: models and tensorboard logs of old runs, OpenAI Codex should not look at that
- `/runs`: models, tensorboard logs and trajectories of the current runs of mobile_robot_rl_2_sac.ipynb, OpenAI Codex should not look at that
- `/urdf`: urdf files of the robot and the maze
- `/other_notebooks`: OpenAI Codex should not look at that

### General Conventions for Agents.md Implementation

- OpenAI Codex should follow the existing code style in each file
- OpenAI Codex should add comments for complex logic

## Pull Request Guidelines for OpenAI Codex

When OpenAI Codex helps create a PR, please ensure it:

1. Includes a clear description of the changes as guided by Agents.md
2. References any related issues that OpenAI Codex is addressing