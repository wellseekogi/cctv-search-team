"""UI 점검용 정적 서버. python http.server는 Range 요청을 지원하지 않아 seek가 안 되므로 Starlette를 쓴다."""
import sys, os
from starlette.applications import Starlette
from starlette.staticfiles import StaticFiles
import uvicorn
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")
app = Starlette()
app.mount("/", StaticFiles(directory=root, html=True))
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765, log_level="warning")
