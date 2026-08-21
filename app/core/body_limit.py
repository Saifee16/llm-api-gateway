import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RequestBodyLimitMiddleware:
    """Buffer HTTP request bodies up to a hard limit before application dispatch.

    The gateway has no request-streaming endpoints. Bounded buffering lets the
    limit apply equally to chunked, missing, and incorrect Content-Length
    requests without allowing the framework to consume an unbounded body.
    """

    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        buffered_messages: list[Message] = []
        body_size = 0
        while True:
            message = await receive()
            buffered_messages.append(message)
            if message["type"] != "http.request":
                break

            body_size += len(message.get("body", b""))
            if body_size > self.max_request_bytes:
                await self._send_too_large(scope, send)
                return
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if buffered_messages:
                return buffered_messages.pop(0)
            return {"type": "http.disconnect"}

        await self.app(scope, replay_receive, send)

    async def _send_too_large(self, scope: Scope, send: Send) -> None:
        state = scope.get("state", {})
        request_id = state.get("request_id")
        payload = json.dumps(
            {
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "REQUEST_TOO_LARGE",
                    "message": "Request body exceeds the configured size limit.",
                },
            }
        ).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(payload)).encode()),
        ]
        if request_id:
            headers.append((b"x-request-id", str(request_id).encode()))
        await send({"type": "http.response.start", "status": 413, "headers": headers})
        await send({"type": "http.response.body", "body": payload})
