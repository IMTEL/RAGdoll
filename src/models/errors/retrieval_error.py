"""Error raised when knowledge retrieval fails for a chat request."""


class RetrievalFailedError(Exception):
    """Knowledge retrieval failed and the agent is configured to fail the request.

    Attributes:
        reason: Short description without secrets, e.g.
            ``"embedding failed: EmbeddingAPIError"``.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)
