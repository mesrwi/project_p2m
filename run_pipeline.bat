@echo off
SETLOCAL EnableDelayedExpansion
REM Conda 환경 이름
set ENV_NAME=venv

REM Conda 설치 경로에 따라 수정 필요
CALL C:\Users\mjk99\miniconda3\Scripts\activate.bat
CALL conda activate %ENV_NAME%

REM 경로 설정
set INFER_PATH=.\infer.py

REM 사용자에게 입력 받기
set /p VIDEO=Enter path to video file (.mp4):
@REM set VIDEO=..\examples\original_clip_1.mp4

REM 인퍼런스 수행
python "%INFER_PATH%"

pause