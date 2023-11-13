import os
import uvicorn
import subprocess

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

if __name__ == "__main__":
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, log_level="info",
                proxy_headers=True, reload=True)
    

