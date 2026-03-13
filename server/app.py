import asyncio
import json
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Poker Game")

STATIC_DIR = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"), headers={"Cache-Control": "no-cache"})


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    game = None
    game_task = None

    async def send_state(state: dict):
        try:
            await ws.send_json(state)
        except Exception:
            pass

    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg["type"] == "new_game":
                # Cancel previous game if running
                if game_task and not game_task.done():
                    game_task.cancel()
                    try:
                        await game_task
                    except (asyncio.CancelledError, Exception):
                        pass

                from server.game import Game
                num_players = max(2, min(8, msg.get("num_players", 6)))
                starting_chips = max(100, msg.get("starting_chips", 1000))
                player_name = msg.get("name", "Player")
                small_blind = max(5, msg.get("small_blind", 10))

                game = Game(
                    human_name=player_name,
                    num_players=num_players,
                    starting_chips=starting_chips,
                    send_callback=send_state,
                    small_blind=small_blind,
                )

                game_task = asyncio.create_task(game.run())

            elif msg["type"] == "action" and game:
                action = msg.get("action", "fold")
                amount = msg.get("amount", 0)
                game.receive_human_action(action, amount)

    except WebSocketDisconnect:
        if game_task and not game_task.done():
            game_task.cancel()
    except Exception:
        if game_task and not game_task.done():
            game_task.cancel()
