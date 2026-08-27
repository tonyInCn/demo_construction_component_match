import sys
import os


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

    print(f"停止端口 {port} 上的服务...")

    if sys.platform == "win32":
        result = os.popen(f"netstat -ano | findstr :{port} | findstr LISTENING").read()
        if result.strip():
            pid = result.strip().split()[-1]
            os.system(f"taskkill /F /PID {pid}")
            print(f"已终止进程 PID: {pid}")
        else:
            print(f"端口 {port} 上没有运行中的服务")
    else:
        os.system(f"lsof -ti:{port} | xargs kill -9 2>/dev/null")
        print(f"已停止端口 {port} 上的服务")


if __name__ == "__main__":
    main()