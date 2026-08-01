from fastapi import FastAPI
import agents

app = FastAPI()

@app.get("/")
def test():
    return {
        "agents_file": agents.__file__,
        "version": getattr(agents, "__version__", "unknown"),
        "has_agent": hasattr(agents, "Agent"),
    }