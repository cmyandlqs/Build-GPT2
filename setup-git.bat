@echo off
REM ========================================
REM  Git 仓库初始化脚本
REM ========================================

echo ========================================
echo   Git 仓库初始化
echo ========================================
echo.

REM 配置用户信息
set /p USERNAME="请输入 GitHub 用户名："
set /p USEREMAIL="请输入 GitHub 邮箱："

git config user.name "%USERNAME%"
git config user.email "%USEREMAIL%"

echo.
echo [√] Git 用户信息已配置
echo.

REM 添加远程仓库
set /p REPO_URL="请输入 GitHub 仓库地址 (https://github.com/YOUR_USERNAME/Build-GPT2.git):"

git remote add origin %REPO_URL%

echo.
echo [√] 远程仓库已添加
echo.

REM 添加文件
echo 正在添加文件...
git add .

echo.
echo [√] 文件已添加
echo.

REM 提交
echo 正在提交...
git commit -m "Initial commit: Build GPT-2 from scratch - Pretraining, Finetuning, Text Generation"

echo.
echo [√] 提交完成
echo.

REM 推送
echo 正在推送到 GitHub...
git branch -M main
git push -u origin main

echo.
echo ========================================
echo   Git 初始化完成!
echo ========================================
echo.

pause
