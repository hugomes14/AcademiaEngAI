"""Run RunningAI with one threaded Gunicorn worker and the local GPU pipeline."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daemon', action='store_true', help='Run in background with logs and PID in outputs/web')
    parser.add_argument('--check-config', action='store_true', help='Validate configuration without starting the server')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    host = os.environ.get('APP_HOST', '127.0.0.1')
    try:
        port = int(os.environ.get('APP_PORT', '5000'))
    except ValueError:
        parser.error('APP_PORT must be an integer')
    if not 1 <= port <= 65535:
        parser.error('APP_PORT must be between 1 and 65535')
    if ':' in host and not host.startswith('['):
        host = '[' + host + ']'
    command = [sys.executable, '-m', 'gunicorn', '--chdir', str(root),
               '--bind', f'{host}:{port}', '--workers', '1', '--threads', '8',
               '--timeout', '120', '--graceful-timeout', '60']
    if args.check_config:
        command.append('--check-config')
    elif args.daemon:
        runtime = root / 'outputs/web'
        runtime.mkdir(parents=True, exist_ok=True)
        command.extend(['--daemon', '--pid', str(runtime/'runningai.pid'),
                        '--access-logfile', str(runtime/'access.log'), '--error-logfile', str(runtime/'error.log')])
        print(f'RunningAI: http://{host}:{port}; logs and PID: {runtime}', flush=True)
    else:
        command.extend(['--access-logfile', '-', '--error-logfile', '-'])
    command.append('app:app')
    os.execv(sys.executable, command)


if __name__ == '__main__':
    main()
