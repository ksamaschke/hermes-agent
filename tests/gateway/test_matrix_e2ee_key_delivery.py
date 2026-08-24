"""Regression tests for Matrix Megolm key delivery after reconnects."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mautrix.types import TrustState, UserID

from plugins.platforms.matrix.adapter import (
    _CryptoStateStore,
    _MatrixStateInspectionError,
    MatrixAdapter,
    _MatrixSessionShareError,
    _build_hermes_olm_machine,
)


ROOM = "!room:example.org"
BOT = "@bot:example.org"
PEER = UserID("@alice:example.org")
DEVICE = "ALICE1"
IDENTITY_KEY = "curve25519:alice"
TARGET = (str(PEER), DEVICE, IDENTITY_KEY)


class _TypedKeyID:
    """Small stand-in for mautrix.types.KeyID.

    It deliberately does not compare equal to a plain string, matching the
    installed Mautrix representation and catching string-key dict lookups.
    """

    def __init__(self, algorithm, key_id):
        self.algorithm = algorithm
        self.key_id = key_id

    def __str__(self):
        return f"{self.algorithm}:{self.key_id}"

    def __hash__(self):
        return hash((self.algorithm, self.key_id))


class FakeSession:
    expired = False
    shared = True
    id = "session-1"

    def __init__(self):
        self.users_shared_with = set()
        self.users_ignored = set()


class FakeStore:
    def __init__(self, session):
        self.session = session
        self.devices = {
            PEER: {
                DEVICE: SimpleNamespace(
                    trust=TrustState.UNVERIFIED,
                    identity_key=IDENTITY_KEY,
                ),
            }
        }
        self.added = []
        self.removed = []

    async def get_devices(self, user_id):
        return self.devices.get(user_id)

    async def get_outbound_group_session(self, room_id):
        return self.session

    async def add_outbound_group_session(self, session):
        self.added.append(session)

    async def remove_outbound_group_sessions(self, rooms):
        self.removed.extend(rooms)


class FakeCrypto:
    def __init__(self, store, recipients=None):
        self.crypto_store = store
        self.state_store = SimpleNamespace(
            is_encrypted=AsyncMock(return_value=True)
        )
        self._fetch_keys_lock = None
        self._fetch_keys = AsyncMock(
            return_value={
                PEER: {
                    DEVICE: SimpleNamespace(identity_key=IDENTITY_KEY),
                }
            }
        )
        self._hermes_last_share_targets = {}
        self.recipients = recipients if recipients is not None else {TARGET}
        self.share_group_session = AsyncMock(side_effect=self._share)

    async def _share(self, room_id, users):
        self._hermes_last_share_targets[str(room_id)] = (
            self.crypto_store.session.id,
            set(self.recipients),
        )
        self.crypto_store.session.shared = True


class FakeClient:
    def __init__(self, crypto):
        self.crypto = crypto
        self.mxid = UserID(BOT)
        self.get_joined_members = AsyncMock(return_value={PEER: object()})
        self.query_keys = AsyncMock(
            return_value=SimpleNamespace(
                failures={},
                device_keys={
                    PEER: {
                        DEVICE: SimpleNamespace(
                            keys={
                                _TypedKeyID("curve25519", DEVICE): IDENTITY_KEY,
                                _TypedKeyID("ed25519", DEVICE): "ed25519:alice",
                            }
                        )
                    }
                },
            )
        )
        self.send_message_event = AsyncMock(return_value="$event:example.org")


def make_adapter(*, recipients=None):
    session = FakeSession()
    store = FakeStore(session)
    crypto = FakeCrypto(store, recipients=recipients)
    client = FakeClient(crypto)
    adapter = object.__new__(MatrixAdapter)
    adapter._client = client
    adapter._user_id = BOT
    adapter._e2ee_mode = "required"
    adapter._encryption = True
    adapter._device_refresh_interval = 300.0
    adapter._device_refresh_ts = {}
    adapter._e2ee_share_lock = asyncio.Lock()
    adapter._matrix_lifecycle_lock = asyncio.Lock()
    adapter._e2ee_room_targets = {}
    adapter._e2ee_machine_active = True
    adapter._matrix_client_connected = True
    adapter._matrix_client_lifecycle_active = True
    adapter._matrix_client_runtime_bound = True
    adapter._matrix_lifecycle_generation = 0
    adapter._e2ee_readiness_tasks = set()
    return adapter, session, store, crypto


@pytest.mark.asyncio
async def test_reconnect_re_shares_persisted_session_to_current_devices():
    adapter, session, store, crypto = make_adapter()

    await adapter._ensure_encrypted_room_ready(ROOM)

    assert session.shared is True
    assert store.added == [session]
    crypto._fetch_keys.assert_awaited_once_with([PEER], include_untracked=True)
    crypto.share_group_session.assert_awaited_once_with(
        ROOM,
        [PEER],
    )
    assert adapter._e2ee_room_targets[ROOM] == ("session-1", {TARGET})


@pytest.mark.asyncio
async def test_failed_share_clears_cache_and_retries():
    adapter, _session, _store, crypto = make_adapter()
    first_share = AsyncMock(side_effect=RuntimeError("send failed"))
    crypto.share_group_session = first_share

    with pytest.raises(RuntimeError, match="send failed"):
        await adapter._ensure_encrypted_room_ready(ROOM)
    assert ROOM not in adapter._e2ee_room_targets
    assert ROOM not in adapter._device_refresh_ts

    second_share = AsyncMock(side_effect=crypto._share)
    crypto.share_group_session = second_share
    await adapter._ensure_encrypted_room_ready(ROOM)
    assert first_share.await_count == 1
    assert second_share.await_count == 1
    assert crypto._fetch_keys.await_count == 2


@pytest.mark.asyncio
async def test_partial_device_refresh_is_not_cached():
    adapter, _session, _store, crypto = make_adapter()
    crypto._fetch_keys = AsyncMock(return_value={})

    with pytest.raises(RuntimeError, match="refresh was incomplete"):
        await adapter._ensure_encrypted_room_ready(ROOM)
    assert ROOM not in adapter._device_refresh_ts

    crypto._fetch_keys = AsyncMock(
        return_value={PEER: {DEVICE: SimpleNamespace(identity_key=IDENTITY_KEY)}}
    )
    await adapter._ensure_encrypted_room_ready(ROOM)
    assert crypto._fetch_keys.await_count == 1


@pytest.mark.asyncio
async def test_zero_refresh_interval_still_refreshes_every_readiness_check():
    adapter, _session, _store, crypto = make_adapter()
    adapter._device_refresh_interval = 0

    await adapter._ensure_encrypted_room_ready(ROOM)
    await adapter._ensure_encrypted_room_ready(ROOM)

    assert adapter._client.query_keys.await_count == 2
    assert crypto._fetch_keys.await_count == 2


@pytest.mark.asyncio
async def test_encrypted_room_without_joined_peers_fails_closed():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._client.get_joined_members = AsyncMock(return_value={})

    with pytest.raises(RuntimeError, match="no joined peer users"):
        await adapter._ensure_encrypted_room_ready(ROOM)


@pytest.mark.asyncio
async def test_deleted_peer_device_is_not_an_e2ee_target():
    adapter, _session, store, _crypto = make_adapter()
    store.devices[PEER][DEVICE].deleted = True

    with pytest.raises(RuntimeError, match="no eligible peer devices"):
        await adapter._ensure_encrypted_room_ready(ROOM)


@pytest.mark.asyncio
async def test_operation_from_previous_lifecycle_is_rejected():
    adapter, _session, _store, _crypto = make_adapter()
    token = adapter._capture_lifecycle_token()
    adapter._matrix_lifecycle_generation += 1

    with pytest.raises(RuntimeError, match="lifecycle changed"):
        await adapter._send_room_event(
            ROOM,
            "m.room.message",
            {"msgtype": "m.text", "body": "stale"},
            lifecycle_token=token,
        )
    adapter._client.send_message_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelled_share_clears_refresh_state():
    adapter, _session, _store, crypto = make_adapter()
    crypto.share_group_session = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await adapter._ensure_encrypted_room_ready(ROOM)
    assert ROOM not in adapter._device_refresh_ts
    assert ROOM not in adapter._e2ee_room_targets


@pytest.mark.asyncio
async def test_initializing_client_blocks_room_send():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._matrix_client_connected = False

    with pytest.raises(RuntimeError, match="still initializing"):
        await adapter._send_room_event(
            ROOM,
            "m.room.message",
            {"msgtype": "m.text", "body": "must not send"},
        )
    adapter._client.send_message_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_clears_readiness_state_and_tasks():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._client = None
    adapter._sync_task = None
    adapter._invite_join_tasks = {}
    adapter._reaction_redaction_tasks = set()
    adapter._e2ee_room_targets[ROOM] = ("session-1", {TARGET})
    adapter._device_refresh_ts[ROOM] = 1.0

    await adapter.disconnect()

    assert not adapter._e2ee_room_targets
    assert not adapter._device_refresh_ts
    assert not adapter._e2ee_readiness_tasks
    assert adapter._matrix_client_lifecycle_active is False


@pytest.mark.asyncio
async def test_unshared_session_never_uses_cached_targets():
    adapter, session, _store, crypto = make_adapter()
    adapter._e2ee_room_targets[ROOM] = (session.id, {TARGET})
    session.shared = False

    await adapter._ensure_encrypted_room_ready(ROOM)

    crypto.share_group_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_session_id_change_forces_a_new_share():
    adapter, session, _store, crypto = make_adapter()
    await adapter._ensure_encrypted_room_ready(ROOM)
    first_share = crypto.share_group_session

    session.id = "session-2"
    second_share = AsyncMock(side_effect=crypto._share)
    crypto.share_group_session = second_share
    await adapter._ensure_encrypted_room_ready(ROOM)

    assert first_share.await_count == 1
    assert second_share.await_count == 1
    assert adapter._e2ee_room_targets[ROOM][0] == "session-2"


@pytest.mark.asyncio
async def test_device_identity_key_change_forces_a_new_share():
    adapter, _session, store, crypto = make_adapter()
    await adapter._ensure_encrypted_room_ready(ROOM)
    first_share = crypto.share_group_session

    store.devices[PEER][DEVICE].identity_key = "curve25519:alice-rotated"
    crypto.recipients = {(str(PEER), DEVICE, "curve25519:alice-rotated")}
    second_share = AsyncMock(side_effect=crypto._share)
    crypto.share_group_session = second_share
    await adapter._ensure_encrypted_room_ready(ROOM)

    assert first_share.await_count == 1
    assert second_share.await_count == 1


@pytest.mark.asyncio
async def test_unencrypted_room_bypasses_e2ee_reconciliation():
    adapter, _session, _store, crypto = make_adapter()
    crypto.state_store.is_encrypted = AsyncMock(return_value=False)

    await adapter._send_room_event(
        ROOM,
        "m.room.message",
        {"msgtype": "m.text", "body": "plain room"},
    )

    crypto._fetch_keys.assert_not_awaited()
    crypto.share_group_session.assert_not_awaited()
    adapter._client.send_message_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_active_crypto_inspection_failure_blocks_optional_send():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._e2ee_mode = "optional"
    adapter._client.get_state_event = AsyncMock(
        side_effect=RuntimeError("Bearer leaked-if-printed")
    )
    adapter._client.crypto.state_store = _CryptoStateStore(
        SimpleNamespace(), set(), adapter._client
    )

    with pytest.raises(_MatrixStateInspectionError):
        await adapter._send_room_event(
            ROOM,
            "m.room.message",
            {"msgtype": "m.text", "body": "must not send"},
        )
    adapter._client.send_message_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_optional_inactive_crypto_blocks_known_encrypted_room():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._e2ee_mode = "optional"
    adapter._e2ee_machine_active = False
    adapter._client.crypto = None
    adapter._client.get_state_event = AsyncMock(
        return_value=SimpleNamespace(
            algorithm="m.megolm.v1.aes-sha2",
            serialize=lambda: {"algorithm": "m.megolm.v1.aes-sha2"},
        )
    )
    adapter._client.state_store = SimpleNamespace(
        get_encryption_info=AsyncMock(
            return_value=SimpleNamespace(algorithm="m.megolm.v1.aes-sha2")
        )
    )

    with pytest.raises(RuntimeError, match="device refresh is unavailable"):
        await adapter._send_room_event(
            ROOM,
            "m.room.message",
            {"msgtype": "m.text", "body": "must not send"},
        )

    adapter._client.send_message_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_zero_recipient_share_is_removed_and_blocks_send():
    adapter, _session, store, crypto = make_adapter(recipients=set())

    with pytest.raises(RuntimeError, match="missed 1 device"):
        await adapter._send_room_event(
            ROOM,
            "m.room.message",
            {"msgtype": "m.text", "body": "must not send"},
        )

    assert store.removed == [ROOM]
    assert ROOM not in adapter._e2ee_room_targets
    adapter._client.send_message_event.assert_not_awaited()


@pytest.mark.asyncio
async def test_room_event_is_not_sent_before_key_delivery_is_verified():
    adapter, _session, _store, crypto = make_adapter()

    event_id = await adapter._send_room_event(
        ROOM,
        "m.room.message",
        {"msgtype": "m.text", "body": "hello"},
    )

    assert event_id == "$event:example.org"
    crypto.share_group_session.assert_awaited_once()
    adapter._client.send_message_event.assert_awaited_once()


class FakeOlmBase:
    def __init__(self, _client, crypto_store, _state_store):
        self.crypto_store = crypto_store
        self._hermes_test_recipients = {TARGET}

    async def _encrypt_and_share_group_session(self, _session, _olm_sessions):
        return None

    async def share_group_session(self, room_id, _users):
        session = SimpleNamespace(room_id=room_id, id=room_id)
        olm_sessions = {
            PEER: {
                DEVICE: (
                    object(),
                    SimpleNamespace(identity_key=IDENTITY_KEY),
                )
            }
        } if self._hermes_test_recipients else {}
        await self._encrypt_and_share_group_session(session, olm_sessions)


@pytest.mark.asyncio
async def test_olm_wrapper_rejects_zero_recipient_share():
    store = SimpleNamespace(remove_outbound_group_sessions=AsyncMock())
    machine_cls = _build_hermes_olm_machine(FakeOlmBase)
    machine = machine_cls(None, store, None)
    machine.log = SimpleNamespace(info=lambda *_args: None)

    await machine.share_group_session(ROOM, [PEER])
    assert machine._hermes_last_share_targets[ROOM] == (ROOM, {TARGET})

    machine._hermes_test_recipients = set()
    with pytest.raises(_MatrixSessionShareError, match="No encrypted to-device recipients"):
        await machine.share_group_session(ROOM, [PEER])
    store.remove_outbound_group_sessions.assert_awaited_once_with([ROOM])


def test_olm_wrapper_preserves_mautrix_room_key_request_handler():
    try:
        from mautrix.crypto import OlmMachine
    except ImportError:
        pytest.skip("mautrix encryption dependencies unavailable")
    if not isinstance(OlmMachine, type) or not hasattr(OlmMachine, "handle_room_key_request"):
        pytest.skip("lightweight mautrix test double has no key-request handler")

    machine_cls = _build_hermes_olm_machine(OlmMachine)
    assert machine_cls.handle_room_key_request is OlmMachine.handle_room_key_request


def test_yaml_refresh_config_bridges_to_environment(monkeypatch):
    from plugins.platforms.matrix.adapter import _apply_yaml_config

    monkeypatch.delenv("MATRIX_DEVICE_REFRESH_SECONDS", raising=False)
    _apply_yaml_config({}, {"device_refresh_seconds": 45})
    assert __import__("os").environ["MATRIX_DEVICE_REFRESH_SECONDS"] == "45"


def test_matrix_exception_summary_redacts_credentials():
    from plugins.platforms.matrix.adapter import _matrix_exception_summary

    summary = _matrix_exception_summary(
        RuntimeError(
            'Bearer syt_secret access_token="also-secret" '
            'https://user:password@example.org/path?token=query-secret'
        )
    )
    assert "syt_secret" not in summary
    assert "also-secret" not in summary
    assert "password@example.org" not in summary
    assert "query-secret" not in summary
    assert "[REDACTED]" in summary


@pytest.mark.asyncio
async def test_reconciliation_failure_does_not_disconnect_gateway():
    adapter, _session, _store, _crypto = make_adapter()
    adapter._reconcile_encrypted_rooms = AsyncMock(return_value=False)

    # The gateway must stay connected so Matrix sync and room-key requests can
    # repair the room. Outbound encrypted sends still call the readiness gate
    # and fail closed until recipients are verified.
    reconciled = await adapter._reconcile_encrypted_rooms_before_ready()

    assert reconciled is False
    adapter._reconcile_encrypted_rooms.assert_awaited_once_with()
    assert adapter._matrix_client_connected is True
