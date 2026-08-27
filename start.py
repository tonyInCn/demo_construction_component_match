import subprocess
import sys
import os


def find_available_port(start_port=8000, max_port=65535):
    import socket
    for port in range(start_port, min(start_port + 100, max_port)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


def kill_port(port):
    if sys.platform == "win32":
        os.system(f"netstat -ano | findstr :{port} | findstr LISTENING > nul")
        result = os.popen(f"netstat -ano | findstr :{port} | findstr LISTENING").read()
        if result.strip():
            pid = result.strip().split()[-1]
            os.system(f"taskkill /F /PID {pid}")
            print(f"已终止端口 {port} 上的进程 (PID: {pid})")
    else:
        os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    print(f"建筑构件识别系统 - 启动脚本")
    print(f"端口: {port}")

    kill_port(port)

    import time
    time.sleep(0.5)

    port = find_available_port(port)
    print(f"使用端口: {port}")

    print("启动 FastAPI 服务...")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "warning",
    ])


if __name__ == "__main__":
    main()