import pytest
from streamlit.testing.v1 import AppTest

def test_greeter_app():
    at = AppTest.from_file("../api/main.py").run(timeout=30)
    assert not at.exception
    # Check that subheaders exist (returns list)
    assert len(at.subheader) > 0
    