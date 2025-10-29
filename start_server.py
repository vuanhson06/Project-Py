# start_server.py
from Backend import app
import uvicorn
import sys

def main():
    # Thử các port theo thứ tự
    ports = [8000, 8001, 8002, 8003, 8004, 8005]
    
    for port in ports:
        try:
            print(f"🚀 Attempting to start on port {port}...")
            uvicorn.run(app, host="0.0.0.0", port=port, reload=True)
            break
        except OSError as e:
            if "Address already in use" in str(e):
                print(f"❌ Port {port} is busy...")
                continue
            else:
                raise e
    else:
        print("❌ All ports are busy! Please close other applications.")
        sys.exit(1)

if __name__ == "__main__":
    main()