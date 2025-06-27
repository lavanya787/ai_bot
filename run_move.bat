@echo off
cd /d "C:\Users\lavan\OneDrive\Desktop\ai_bot"
echo Activating virtualenv... >> logs\run_move_log.txt
call menv\Scripts\activate >> logs\run_move_log.txt 2>&1
echo Running script... >> logs\run_move_log.txt
python auto_domain_mover.py >> logs\run_move_log.txt 2>&1
echo Script finished at %time% >> logs\run_move_log.txt
pause
