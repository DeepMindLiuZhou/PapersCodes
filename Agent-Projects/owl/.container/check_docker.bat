@echo off
chcp 65001 >nul
echo 检查Docker环境...
echo Checking Docker environment...

REM 检查Docker是否安装
REM Check if Docker is installed
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker未安装
    echo Error: Docker not installed
    echo 在Windows上安装Docker的方法:
    echo How to install Docker on Windows:
    echo 1. 访问 https://docs.docker.com/desktop/install/windows-install/ 下载Docker Desktop
    echo 1. Visit https://docs.docker.com/desktop/install/windows-install/ to download Docker Desktop
    echo 2. 安装并启动Docker Desktop
    echo 2. Install and start Docker Desktop
    pause
    exit /b 1
)

echo Docker已安装
echo Docker is installed

REM 检查Docker Compose是否安装
REM Check if Docker Compose is installed
where docker-compose >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 警告: Docker-Compose未找到，尝试使用新的docker compose命令
    echo Warning: Docker-Compose not found, trying to use new docker compose command
    docker compose version >nul 2>nul
    if %ERRORLEVEL% NEQ 0 (
        echo 错误: Docker Compose未安装
        echo Error: Docker Compose not installed
        echo Docker Desktop for Windows应该已包含Docker Compose
        echo Docker Desktop for Windows should already include Docker Compose
        echo 请确保Docker Desktop已正确安装
        echo Please make sure Docker Desktop is properly installed
        pause
        exit /b 1
    ) else (
        echo 使用新的docker compose命令
        echo Using new docker compose command
        set COMPOSE_CMD=docker compose
    )
) else (
    echo Docker-Compose已安装
    echo Docker-Compose is installed
    set COMPOSE_CMD=docker-compose
)

REM 检查Docker是否正在运行
REM Check if Docker is running
docker info >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo 错误: Docker未运行
    echo Error: Docker not running
    echo 请启动Docker Desktop应用程序
    echo Please start Docker Desktop application
    pause
    exit /b 1
)

echo Docker正在运行
echo Docker is running

REM 检查是否有.env文件
REM Check if .env file exists
if not exist "..\owl\.env" (
    echo 警告: 未找到owl\.env文件
    echo Warning: owl\.env file not found
    echo 请运行以下命令创建环境变量文件
    echo Please run the following command to create environment variable file:
    echo copy ..\owl\.env_template ..\owl\.env
    echo 然后编辑owl\.env文件，填写必要的API密钥
    echo Then edit owl\.env file and fill in necessary API keys
) else (
    echo 环境变量文件已存在
    echo Environment variable file exists
)

echo 所有检查完成，您的系统已准备好构建和运行OWL项目的Docker容器
echo All checks completed, your system is ready to build and run OWL project Docker container
echo 请运行以下命令构建Docker镜像:
echo Please run the following command to build Docker image:
echo %COMPOSE_CMD% build

pause