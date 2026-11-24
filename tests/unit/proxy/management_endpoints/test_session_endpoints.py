"""
Tests for POST /session/logout and revoke_ui_session_keys
(litellm/proxy/management_endpoints/session_endpoints.py).
"""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.cookies import SimpleCookie
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
import time_machine
from fastapi import HTTPException, Request, Response
from pydantic import SecretStr

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_VerificationToken, UserAPIKeyAuth, hash_token
from litellm.proxy.management_endpoints.session_endpoints import (
    SessionSyncRequest,
    revoke_ui_session_keys,
    session_logout,
    sync_session_cookie,
)

HASHED_TOKEN = "hashed-session-token"
USER_ID = "user-123"
SIGNING_KEY: Final = "test-session-signing-key-at-least-32-characters"
SESSION_KEY: Final = "sk-test-dashboard-session"
SESSION_EXPIRES: Final = int(datetime(2026, 11, 25, 18, 55, 19, tzinfo=timezone.utc).timestamp())


def _sync_request(site: str = "same-origin", headers: tuple[tuple[bytes, bytes], ...] = ()) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "https",
            "path": "/session/sync",
            "headers": ((b"host", b"gateway.example.test"), (b"sec-fetch-site", site.encode()), *headers),
        }
    )


def _signed_session(**overrides: object) -> str:
    return jwt.encode(
        {
            "user_id": USER_ID,
            "key": SESSION_KEY,
            "exp": SESSION_EXPIRES,
            "login_method": "username_password",
            **overrides,
        },
        SIGNING_KEY,
        algorithm="HS256",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("login_method", ("username_password", "sso"))
@time_machine.travel("2026-11-24T18:55:19Z", tick=False)
async def test_sync_restores_the_same_session_with_its_original_expiry(login_method: str) -> None:
    token: Final = _signed_session(login_method=login_method)
    with patch("litellm.proxy.proxy_server.master_key", SIGNING_KEY):
        response: Final = await sync_session_cookie(
            _sync_request(),
            SessionSyncRequest(token=SecretStr(token)),
            _ui_session_caller(hash_token(SESSION_KEY)),
        )
    cookie: Final[SimpleCookie[str]] = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert cookie["token"].value == token
    assert parsedate_to_datetime(cookie["token"]["expires"]).timestamp() == SESSION_EXPIRES
    assert cookie["token"]["httponly"]
    assert cookie["token"]["secure"]
    assert cookie["token"]["samesite"] == "lax"
    assert cookie["token"]["path"] == "/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    (
        {"exp": 0},
        {"key": "sk-different-session"},
        {"user_id": "different-user"},
        {"login_method": "api_key"},
        {"type": "byok_session"},
    ),
)
@time_machine.travel("2026-11-24T18:55:19Z", tick=False)
async def test_sync_refuses_a_token_that_does_not_represent_the_authenticated_session(
    overrides: dict[str, object],
) -> None:
    with patch("litellm.proxy.proxy_server.master_key", SIGNING_KEY), pytest.raises(HTTPException) as error:
        await sync_session_cookie(
            _sync_request(),
            SessionSyncRequest(token=SecretStr(_signed_session(**overrides))),
            _ui_session_caller(hash_token(SESSION_KEY)),
        )
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_sync_refuses_cross_origin_session_fixation() -> None:
    with pytest.raises(HTTPException) as error:
        await sync_session_cookie(
            _sync_request("cross-site"),
            SessionSyncRequest(token=SecretStr(_signed_session())),
            _ui_session_caller(hash_token(SESSION_KEY)),
        )
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "token",
    (
        "not-a-jwt",
        jwt.encode({"exp": SESSION_EXPIRES}, "different-signing-key-at-least-32-characters"),
        jwt.encode({"user_id": USER_ID, "key": SESSION_KEY, "login_method": "sso"}, SIGNING_KEY),
    ),
)
@time_machine.travel("2026-11-24T18:55:19Z", tick=False)
async def test_sync_rejects_unsigned_forged_or_unbounded_tokens(token: str) -> None:
    with patch("litellm.proxy.proxy_server.master_key", SIGNING_KEY), pytest.raises(HTTPException) as error:
        await sync_session_cookie(
            _sync_request(), SessionSyncRequest(token=SecretStr(token)), _ui_session_caller(hash_token(SESSION_KEY))
        )
    assert error.value.status_code == 401


@pytest.mark.asyncio
@time_machine.travel("2026-11-24T18:55:19Z", tick=False)
async def test_sync_cannot_turn_an_api_key_into_a_dashboard_cookie() -> None:
    with patch("litellm.proxy.proxy_server.master_key", SIGNING_KEY), pytest.raises(HTTPException) as error:
        await sync_session_cookie(
            _sync_request(),
            SessionSyncRequest(token=SecretStr(_signed_session())),
            UserAPIKeyAuth(token=hash_token(SESSION_KEY), user_id=USER_ID),
        )
    assert error.value.status_code == 403


@pytest.mark.asyncio
@time_machine.travel("2026-11-24T18:55:19Z", tick=False)
async def test_sync_preserves_experimental_sessions_using_their_authenticated_custom_header() -> None:
    token: Final = _signed_session(key="encrypted-session", auth_header_name="X-Session-Key")
    with patch("litellm.proxy.proxy_server.master_key", SIGNING_KEY):
        response: Final = await sync_session_cookie(
            _sync_request(headers=((b"x-session-key", b"Bearer encrypted-session"),)),
            SessionSyncRequest(token=SecretStr(token)),
            _ui_session_caller("ui-token"),
        )
    assert response.status_code == 200
    assert f"token={token}" in response.headers["set-cookie"]


def _session_row(token: str = HASHED_TOKEN, user_id: str = USER_ID) -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=user_id)


def _make_prisma(
    find_unique_row: LiteLLM_VerificationToken | None = None,
    find_many_rows: list[LiteLLM_VerificationToken] | None = None,
) -> MagicMock:
    prisma = MagicMock()
    table = prisma.db.litellm_verificationtoken
    table.find_unique = AsyncMock(return_value=find_unique_row)
    table.find_many = AsyncMock(return_value=find_many_rows or [])
    table.delete_many = AsyncMock(return_value=1)
    return prisma


def _ui_session_caller(token: str | None = HASHED_TOKEN) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token=token, team_id=UI_SESSION_TOKEN_TEAM_ID, user_id=USER_ID)


def _patched_globals(prisma):
    return (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", prisma
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.proxy_logging_obj", None
        ),
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.user_api_key_cache", MagicMock()
        ),
    )


@pytest.mark.asyncio
async def test_session_logout_revokes_presented_session():
    prisma = _make_prisma(find_unique_row=_session_row())
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session revoked."
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": HASHED_TOKEN}
    # Audit record persisted before the row is gone.
    persist_mock.assert_awaited_once()
    assert persist_mock.await_args.kwargs["keys"][0].token == HASHED_TOKEN
    # Cache evicted + broadcast even on the delete path.
    evict_mock.assert_awaited_once()
    assert tuple(evict_mock.await_args.kwargs["hashed_tokens"]) == (HASHED_TOKEN,)


@pytest.mark.asyncio
async def test_session_logout_clears_token_cookie():
    prisma = _make_prisma(find_unique_row=_session_row())
    fastapi_response = Response()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        await session_logout(
            response=fastapi_response,
            user_api_key_dict=_ui_session_caller(),
        )

    set_cookie_headers = [v.decode() for k, v in fastapi_response.raw_headers if k == b"set-cookie"]
    assert any(h.startswith('token="";') or h.startswith("token=;") for h in set_cookie_headers)


@pytest.mark.asyncio
async def test_session_logout_refuses_non_ui_session_key():
    """The endpoint must not become a generic key-deletion oracle: a normal
    virtual key (no UI team id) is refused outright."""
    prisma = _make_prisma()
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=UserAPIKeyAuth(token=HASHED_TOKEN, team_id="some-real-team", user_id=USER_ID),
            )

    assert exc_info.value.status_code == 403
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_session_logout_is_idempotent_when_row_already_gone():
    prisma = _make_prisma(find_unique_row=None)
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        response = await session_logout(
            response=Response(),
            user_api_key_dict=_ui_session_caller(),
        )

    assert response.message == "Session already revoked."
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()
    # The cache entry may outlive the row; evict regardless.
    evict_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_logout_requires_db():
    p2 = patch("litellm.proxy.proxy_server.proxy_logging_obj", None)
    p3 = patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock())
    with (
        patch(  # test-quality-ok: endpoint reads proxy_server module globals; no injection seam
            "litellm.proxy.proxy_server.prisma_client", None
        ),
        p2,
        p3,
    ):
        with pytest.raises(HTTPException) as exc_info:
            await session_logout(
                response=Response(),
                user_api_key_dict=_ui_session_caller(),
            )

    assert exc_info.value.status_code == 500


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_revokes_all_and_broadcasts():
    rows = [_session_row(token="t1"), _session_row(token="t2"), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    persist_mock = AsyncMock()
    evict_mock = AsyncMock()
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            persist_mock,
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            evict_mock,
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 3
    find_kwargs = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs
    assert find_kwargs["where"] == {"user_id": USER_ID, "team_id": UI_SESSION_TOKEN_TEAM_ID}
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t2", "t3"]}}
    persist_mock.assert_awaited_once()
    evict_mock.assert_awaited_once()
    assert evict_mock.await_args.kwargs["hashed_tokens"] == ["t1", "t2", "t3"]


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_keeps_callers_session():
    rows = [_session_row(token="t1"), _session_row(token=HASHED_TOKEN), _session_row(token="t3")]
    prisma = _make_prisma(find_many_rows=rows)
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.delete_cache_key_objects",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
            keep_hashed_token=HASHED_TOKEN,
        )

    assert revoked == 2
    delete_kwargs = prisma.db.litellm_verificationtoken.delete_many.call_args.kwargs
    assert delete_kwargs["where"] == {"token": {"in": ["t1", "t3"]}}


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_when_no_sessions():
    prisma = _make_prisma(find_many_rows=[])
    p1, p2, p3 = _patched_globals(prisma)

    with p1, p2, p3:
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0
    prisma.db.litellm_verificationtoken.delete_many.assert_not_called()


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_failure_is_swallowed():
    """The password write has already committed when this runs; a revocation
    failure must not fail the caller's request."""
    prisma = _make_prisma(find_many_rows=[_session_row(token="t1")])
    prisma.db.litellm_verificationtoken.delete_many = AsyncMock(side_effect=RuntimeError("db down"))
    p1, p2, p3 = _patched_globals(prisma)

    with (
        p1,
        p2,
        p3,
        patch(
            "litellm.proxy.management_endpoints.session_endpoints.persist_deleted_verification_tokens",
            AsyncMock(),
        ),
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0


@pytest.mark.asyncio
async def test_revoke_ui_session_keys_noop_without_db():
    with patch(  # test-quality-ok: helper reads proxy_server module globals; no injection seam
        "litellm.proxy.proxy_server.prisma_client", None
    ):
        revoked = await revoke_ui_session_keys(
            user_id=USER_ID,
            user_api_key_dict=_ui_session_caller(),
        )

    assert revoked == 0
