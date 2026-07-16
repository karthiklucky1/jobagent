"""Regression: autofill AnswerMemory reads/writes must be scoped by the owning
user_id, or one tenant's cached free-text answers (salary, essays) leak into
another tenant's forms (app/autofill/agent.py — the #2 critical finding).
"""
import contextvars

import pytest
from sqlmodel import select, delete

from app.db.init_db import get_session
from app.db.models import AnswerMemory
import app.autofill.agent as af


@pytest.fixture(autouse=True)
def _clean():
    with get_session() as s:
        s.exec(delete(AnswerMemory).where(AnswerMemory.label_normalized == "regr salary"))
        s.commit()
    yield
    with get_session() as s:
        s.exec(delete(AnswerMemory).where(AnswerMemory.label_normalized == "regr salary"))
        s.commit()


def test_current_owner_maps_local_and_unset_to_none():
    # Run inside a fresh context so the ContextVar default is observed.
    def _check():
        assert af._current_owner() is None            # unset
        af._autofill_owner.set("local")
        assert af._current_owner() is None            # single-user
        af._autofill_owner.set("user-A")
        assert af._current_owner() == "user-A"
    contextvars.copy_context().run(_check)


def test_scoped_query_isolates_tenants():
    with get_session() as s:
        s.add(AnswerMemory(user_id="user-A", label_normalized="regr salary",
                           label_original="Salary", answer="A-150k"))
        s.add(AnswerMemory(user_id="user-B", label_normalized="regr salary",
                           label_original="Salary", answer="B-90k"))
        s.commit()

    with get_session() as s:
        a = s.exec(af._scope_answer_memory(
            select(AnswerMemory).where(AnswerMemory.label_normalized == "regr salary"),
            "user-A")).first()
        b = s.exec(af._scope_answer_memory(
            select(AnswerMemory).where(AnswerMemory.label_normalized == "regr salary"),
            "user-B")).first()
    assert a.answer == "A-150k"           # A never sees B's answer
    assert b.answer == "B-90k"


def test_none_owner_only_matches_legacy_null_rows():
    with get_session() as s:
        s.add(AnswerMemory(user_id="user-A", label_normalized="regr salary",
                           label_original="Salary", answer="A-150k"))
        s.commit()
    with get_session() as s:
        row = s.exec(af._scope_answer_memory(
            select(AnswerMemory).where(AnswerMemory.label_normalized == "regr salary"),
            None)).first()
    assert row is None                    # a NULL-owner read must NOT see A's row
