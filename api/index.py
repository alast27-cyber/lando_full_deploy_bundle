from app.main import app

# Explicit handler alias helps some runtimes/tools discover the ASGI/WSGI entrypoint.
handler = app
