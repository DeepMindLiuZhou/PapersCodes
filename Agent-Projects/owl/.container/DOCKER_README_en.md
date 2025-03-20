# OWL Project Docker Usage Guide

This document provides detailed instructions on how to run the OWL project using Docker.

## Prerequisites

• Install [Docker](https://docs.docker.com/get-docker/)
• Install [Docker Compose](https://docs.docker.com/compose/install/) (recommended v2.x version)
• Obtain necessary API keys (OpenAI API, etc.)

## Technical Notes

This Docker configuration uses the following technologies to ensure the OWL project runs smoothly in containers:

• **Xvfb**: Virtual framebuffer, used to simulate an X server in a headless environment
• **Playwright**: Used for browser automation, configured in headless mode
• **Shared Memory**: Increased shared memory size to improve browser performance
• **BuildKit**: Uses Docker BuildKit to accelerate the build process
• **Cache Optimization**: Uses persistent volumes to cache pip and Playwright dependencies
• **Cross-Platform Compatibility**: Provides scripts for both Windows and macOS/Linux

## Docker Compose Version Notes

The docker-compose.yml file used in this project is compatible with Docker Compose v2.x. If you are using an older Docker Compose v1.x version, you may need to manually add the version number:

```yaml
version: '3'

services:
  # ...rest of the configuration remains unchanged
```

## Quick Start

### 0. Check Environment

First, run the check script to ensure your environment is ready:

#### Check on macOS/Linux

```bash
# First, add execute permissions to the script
chmod +x check_docker.sh

# Run the check script
./check_docker.sh
```

#### Check on Windows

```cmd
check_docker.bat
```

If the check script finds any issues, please follow the prompts to fix them.

### 1. Configure Environment Variables

Copy the environment variable template file and fill in the necessary API keys:

```bash
cp owl/.env_template owl/.env
```

Then edit the `owl/.env` file and fill in the necessary API keys, for example:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
SEARCH_ENGINE_ID=your_search_engine_id
```

### 2. Quick Build Docker Image

#### Build on macOS/Linux

Use the provided shell script to speed up the Docker image build:

```bash
# First, add execute permissions to the script
chmod +x build_docker.sh

# Run the build script
./build_docker.sh
```

#### Build on Windows

Use the provided batch file:

```cmd
build_docker.bat
```

Or build and start using the standard method:

```bash
# Use BuildKit to accelerate the build
set DOCKER_BUILDKIT=1
set COMPOSE_DOCKER_CLI_BUILD=1
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1

# Start the container
docker-compose up -d
```

### 3. Interactive Use of the Container

After the container starts, it will automatically enter an interactive shell environment and display a welcome message and a list of available scripts:

```bash
# Enter the container (if not automatically entered)
docker-compose exec owl bash
```

Inside the container, you can directly run any available script:

```bash
# Run the default script
xvfb-python run.py

# Run the DeepSeek example
xvfb-python run_deepseek_example.py

# Run the script and pass query parameters
xvfb-python run.py "What is artificial intelligence?"
```

### 4. Run Queries Using External Scripts

#### Run on macOS/Linux

```bash
# First, add execute permissions to the script
chmod +x run_in_docker.sh

# Default to using the run.py script
./run_in_docker.sh "your question"

# Specify a particular script
./run_in_docker.sh run_deepseek_example.py "your question"
```

#### Run on Windows

```cmd
REM Default to using the run.py script
run_in_docker.bat "your question"

REM Specify a particular script
run_in_docker.bat run_deepseek_example.py "your question"
```

**Available Scripts**:
• `run.py` - Default script, uses OpenAI GPT-4o model
• `run_deepseek_example.py` - Uses the DeepSeek model
• `run_gaia_roleplaying.py` - GAIA benchmark script

## Directory Mounts

The Docker Compose configuration has set up the following mount points:

• `./owl/.env:/app/owl/.env`: Mounts the environment variable file for easy modification of API keys
• `./data:/app/data`: Mounts the data directory for storing and accessing data files
• `playwright-cache`: Persistent volume for caching Playwright browsers
• `pip-cache`: Persistent volume for caching pip packages

## Environment Variables

You can set environment variables in two ways:

1. Modify the `owl/.env` file
2. Add environment variables in the `environment` section of the `docker-compose.yml` file

## Build Optimization

This Docker configuration includes several build optimizations:

1. **Use of Domestic Mirror Sources**: Uses Tsinghua University mirror sources to accelerate pip package downloads
2. **Layer Optimization**: Reduces the number of layers in the Dockerfile to improve build efficiency
3. **Cache Utilization**:
   • Enables pip caching to avoid repeated dependency downloads
   • Uses Docker BuildKit inline caching
   • Arranges Dockerfile instructions to maximize cache utilization
4. **BuildKit**: Enables Docker BuildKit to accelerate builds
5. **Persistent Caching**:
   • Uses Docker volumes to cache pip packages (`pip-cache`)
   • Uses Docker volumes to cache Playwright browsers (`playwright-cache`)
   • Local cache directory (`.docker-cache`)

### Cache Cleanup

If you need to clean the cache, you can use the following commands:

```bash
# Clean Docker build cache
docker builder prune

# Clean Docker volumes (will delete all unused volumes, including cache volumes)
docker volume prune

# Clean local cache directory
rm -rf .docker-cache
```

## Cross-Platform Compatibility

This project provides scripts for different operating systems:

1. **Check Scripts**:
   • `check_docker.sh` (macOS/Linux): Checks the Docker environment
   • `check_docker.bat` (Windows): Checks the Docker environment

2. **Build Scripts**:
   • `build_docker.sh` (macOS/Linux): Builds the Docker image
   • `build_docker.bat` (Windows): Builds the Docker image

3. **Run Scripts**:
   • `run_in_docker.sh` (macOS/Linux): Runs scripts in the Docker container
   • `run_in_docker.bat` (Windows): Runs scripts in the Docker container

These scripts automatically detect the operating system type and use appropriate commands.

## Troubleshooting

### Container Fails to Start

Check the logs for more information:

```bash
docker-compose logs
```

### API Key Issues

Ensure that you have correctly set all necessary API keys in the `owl/.env` file.

### Docker Compose Warnings

If you see a warning about the `version` attribute being obsolete:

```
WARN[0000] docker-compose.yml: the attribute `version` is obsolete
```

This is because you are using Docker Compose v2.x, which no longer requires an explicit version number. We have removed this attribute from the configuration file, so you should no longer see this warning.

### Browser-Related Issues

If you encounter browser-related issues, try the following solutions:

1. Ensure that you are using the `xvfb-python` command to run Python scripts in the Docker container
2. Check that Xvfb and related dependencies are correctly installed
3. Increase the shared memory size (set to 2GB in docker-compose.yml)

### Slow Build Speed

If the build speed is slow, try the following solutions:

1. Ensure that Docker BuildKit is enabled (`DOCKER_BUILDKIT=1`)
2. Ensure that pip caching is enabled (configured in docker-compose.yml)
3. Use the `--build-arg BUILDKIT_INLINE_CACHE=1` parameter when building (configured in the build script)
4. If this is the first build, downloading dependencies may take some time, but subsequent builds will be faster

### Windows-Specific Issues

If you encounter issues on Windows:

1. Ensure that you are running the Command Prompt or PowerShell with administrator privileges
2. If you encounter path issues, try using forward slashes (/) instead of backslashes (\)
3. If you encounter Docker Compose command issues, try using `docker compose` (without the hyphen)

### Insufficient Memory

If you encounter insufficient memory issues, you can adjust resource limits in the `docker-compose.yml` file:

```yaml
services:
  owl:
    # Other configurations...
    deploy:
      resources:
        limits:
          cpus: '4'  # Increase CPU cores
          memory: 8G  # Increase memory limit
```

## Custom Docker Image

If you need to customize the Docker image, modify the `Dockerfile` file and then rebuild:

```bash
# macOS/Linux
./build_docker.sh

# Windows
build_docker.bat
```