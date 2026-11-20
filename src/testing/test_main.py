import pytest
from streamlit.testing.v1 import AppTest
#----------- streamlit testing-------------------
def test_greeter_app():
    at = AppTest.from_file("main.py").run()
    assert not at.exception
    assert at.subheader.value == "Model overview"
    