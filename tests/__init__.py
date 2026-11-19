"""Unit tests of the goal-loop plugin. No test may touch the Hermes that runs the tests: the
follow-up run seams of resume.py get a fake job maker and a temporary Hermes home (a test that
needs a job sets its own fake)."""
import tempfile

from goal_loop import resume

resume.CREATE_JOB = lambda **kwargs: {}
resume.HERMES_HOME = tempfile.mkdtemp(prefix="gl-home-")
