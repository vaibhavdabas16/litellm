"""
UI session cookies and revocation.

POST /session/sync restores the browser cookie without extending the session.
POST /session/logout — revoke the UI session key this request authenticated with.
revoke_ui_session_keys — revoke every UI session key a user holds (password writes).

Logging out of the dashboard was purely client-side (cookies cleared, redirect);
the DB-backed virtual key minted at login stayed valid until
LITELLM_UI_SESSION_DURATION elapsed, so a captured token kept working access
after logout, and changing a password did not invalidate existing sessions.

Deliberately NOT reusing /key/delete: its `can_modify_verification_token`
ownership checks can reject low-privilege roles, and a self-revoke endpoint
that takes no body cannot be aimed at other keys.
"""

from datetime import datetime, timezone
from secrets import compare_digest
from typing import TYPE_CHECKING, Annotated, Final, Literal, cast

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import (
    CommonProxyErrors,
    HTTPExceptionErrorDetail,
    LiteLLM_VerificationToken,
    SessionLogoutResponse,
    UserAPIKeyAuth,
    hash_token,
)
from litellm.proxy.auth.auth_checks import delete_cache_key_objects
from litellm.proxy.auth.ip_address_utils import IPAddressUtils
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.key_management_endpoints import (  # noqa: F401  # legacy module exports
    _persist_deleted_verification_tokens,  # pyright: ignore[reportPrivateUsage,reportUnusedImport]  # backwards-compatible package export
    persist_deleted_verification_tokens,
)
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)

if TYPE_CHECKING:
    from prisma import types as prisma_types

router: Final = APIRouter()

_TOKEN_LIST: Final = TypeAdapter(list[str])


class SessionSyncRequest(BaseModel):
    model_config = ConfigDict(frozen=True)
    token: SecretStr = Field(description="The signed dashboard JWT for the authenticated UI session.")


class SessionSyncResponse(BaseModel):
    model_config = ConfigDict(frozen=True)
    expires_at: int = Field(description="The existing session's expiration as a Unix timestamp, without renewal.")


class _DashboardSessionClaims(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    exp: int
    login_method: Literal["sso", "username_password"]
    auth_header_name: str = "Authorization"
    type: None = None


@router.post("/session/sync", tags=["UI Session"], response_model=SessionSyncResponse)
async def sync_session_cookie(
    request: Request,
    body: SessionSyncRequest,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> Response:
    """Restore the browser cookie for an existing dashboard session without renewing it."""
    from litellm.proxy.proxy_server import master_key

    scheme: Final = "https" if IPAddressUtils.is_request_https(request) else "http"
    site: Final = request.headers.get("sec-fetch-site")
    same_origin: Final = (
        site == "same-origin"
        if site is not None
        else request.headers.get("origin") == f"{scheme}://{request.headers.get('host')}"
    )
    if not same_origin:
        raise HTTPException(403, "Session cookies must be restored from the LiteLLM dashboard.")
    if not master_key or user_api_key_dict.team_id != UI_SESSION_TOKEN_TEAM_ID:
        raise HTTPException(403, "Only UI sessions can restore a dashboard cookie.")
    token: Final = body.token.get_secret_value()
    try:
        claims: Final = _DashboardSessionClaims.model_validate(
            jwt.decode(token, master_key, algorithms=["HS256"], options={"require": ["exp"]})
        )
    except (jwt.InvalidTokenError, ValidationError):
        raise HTTPException(401, "Invalid dashboard session.") from None
    same_key: Final = hash_token(claims.key) == user_api_key_dict.token or (
        user_api_key_dict.token == "ui-token"
        and compare_digest(request.headers.get(claims.auth_header_name, ""), f"Bearer {claims.key}")
    )
    if claims.user_id != user_api_key_dict.user_id or not same_key:
        raise HTTPException(401, "The dashboard token must belong to the authenticated session.")
    response: Final = JSONResponse(
        content=SessionSyncResponse(expires_at=claims.exp).model_dump(), headers={"Cache-Control": "no-store"}
    )
    response.set_cookie(
        "token",
        token,
        expires=datetime.fromtimestamp(claims.exp, tz=timezone.utc),
        secure=scheme == "https",
        httponly=True,
        samesite="lax",
    )
    return response


def _error_detail(message: str) -> HTTPExceptionErrorDetail:
    detail: Final[HTTPExceptionErrorDetail] = {"error": message}
    return detail


async def revoke_ui_session_keys(
    user_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    keep_hashed_token: str | None = None,
    litellm_changed_by: str | None = None,
) -> int:
    """Revoke every UI session key belonging to ``user_id``, except
    ``keep_hashed_token`` (the caller's own session on a self-service password
    change; the other password-write paths revoke all).

    Best-effort: the password write this runs after has already committed, so a
    revocation failure is logged loudly rather than failing the request — the
    unrevoked keys still expire at LITELLM_UI_SESSION_DURATION.

    Returns the number of sessions revoked.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        return 0

    try:
        where_user_sessions: Final[prisma_types.LiteLLM_VerificationTokenWhereInput] = {
            "user_id": user_id,
            "team_id": UI_SESSION_TOKEN_TEAM_ID,
        }
        rows: Final = cast(  # cast-ok: find_many returns prisma rows shaped like the pydantic model
            "tuple[LiteLLM_VerificationToken, ...]",
            tuple(await VerificationTokenRepository(prisma_client).table.find_many(where=where_user_sessions)),
        )
        revoked_rows: Final = tuple(row for row in rows if row.token is not None and row.token != keep_hashed_token)
        if not revoked_rows:
            return 0
        revoked_tokens: Final = _TOKEN_LIST.validate_python(tuple(row.token for row in revoked_rows))

        await persist_deleted_verification_tokens(
            keys=revoked_rows,
            prisma_client=prisma_client,
            user_api_key_dict=user_api_key_dict,
            litellm_changed_by=litellm_changed_by,
        )
        where_revoked: Final[prisma_types.LiteLLM_VerificationTokenWhereInput] = {"token": {"in": revoked_tokens}}
        await VerificationTokenRepository(prisma_client).table.delete_many(where=where_revoked)
        await delete_cache_key_objects(
            hashed_tokens=revoked_tokens,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        verbose_proxy_logger.info(
            "Revoked %s UI session key(s) for user_id=%s after password change",
            len(revoked_tokens),
            user_id,
        )
        return len(revoked_tokens)
    except Exception:  # noqa: BLE001  # the password write committed; revocation must not undo that
        verbose_proxy_logger.exception(
            "Failed to revoke UI session keys for user_id=%s; existing sessions remain valid until they expire",
            user_id,
        )
        return 0


@router.post(
    "/session/logout",
    tags=("UI Session",),
)
async def session_logout(
    response: Response,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
) -> SessionLogoutResponse:
    """
    Revoke the UI session key this request authenticated with.

    Only accepts UI session keys (minted by dashboard login); any other
    credential is refused, so this can never be used to delete arbitrary keys.
    Revokes only the presented session, not the user's other sessions.
    Idempotent: logging out an already-revoked session succeeds.
    """
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail=_error_detail(CommonProxyErrors.db_not_connected_error.value),
        )

    if user_api_key_dict.team_id != UI_SESSION_TOKEN_TEAM_ID:
        raise HTTPException(
            status_code=403,
            detail=_error_detail("Only UI session tokens can be revoked through this endpoint."),
        )

    hashed_token: Final = user_api_key_dict.token
    revoked = False
    if hashed_token is not None:
        where_token: Final[prisma_types.LiteLLM_VerificationTokenWhereUniqueInput] = {"token": hashed_token}
        row: Final = await VerificationTokenRepository(prisma_client).table.find_unique(where=where_token)
        # A missing row means the session is already revoked (or an
        # EXPERIMENTAL_UI_LOGIN blob token); logout is idempotent either way.
        if row is not None:
            caller_row: Final = cast(  # cast-ok: find_unique returns a prisma row shaped like the pydantic model
                "LiteLLM_VerificationToken", row
            )
            await persist_deleted_verification_tokens(
                keys=(caller_row,),
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
            )
            await VerificationTokenRepository(prisma_client).table.delete_many(where=where_token)
            revoked = True
        await delete_cache_key_objects(
            hashed_tokens=(hashed_token,),
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )

    # The server set this cookie at login (set_session_token_cookie); clear it
    # here too so logout works even if the client-side clear is skipped.
    response.delete_cookie("token")
    return SessionLogoutResponse(
        message="Session revoked." if revoked else "Session already revoked.",
    )
