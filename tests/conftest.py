import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_d = tempfile.mkdtemp(); os.environ["PRICESENSE_DB"] = os.path.join(_d, "test.db")
