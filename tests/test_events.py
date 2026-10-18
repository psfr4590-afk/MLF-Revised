import logging

from ui.core import events


def test_publish_isolates_failed_callback_and_logs(caplog):
    events._subscribers.clear()
    seen = []

    def broken(data):
        raise RuntimeError("callback exploded")

    def healthy(data):
        seen.append(data)

    events.subscribe("test", broken)
    events.subscribe("test", healthy)

    with caplog.at_level(logging.ERROR, logger="ui.core.events"):
        events.publish("test", {"ok": True})

    assert seen == [{"ok": True}]
    assert "UI event callback failed" in caplog.text
    assert "callback exploded" in caplog.text
    events._subscribers.clear()


def test_unsubscribe_missing_callback_is_non_fatal():
    events._subscribers.clear()
    events.unsubscribe("missing", lambda _data: None)
