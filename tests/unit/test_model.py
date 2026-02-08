import os
import sys

import numpy as np
import onnxruntime as ort
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nyc_taxi_model.onnx")


@pytest.fixture(scope="module")
def shared_session():
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"Model file not found! Path: {MODEL_PATH}")
    try:
        session = ort.InferenceSession(MODEL_PATH)
        return session
    except Exception as e:
        pytest.fail(f"Model failed to load during setup: {str(e)}")


class TestModelArtifact:
    """
    Model Artifact Tests:
    Tests the existence, loadability, and predictive ability of the trained .onnx model file.
    """

    def test_model_file_exists(self):
        """
        Test: Does the model file physically exist on the disk?
        Checking path is cheap, so we can keep this separate or rely on fixture.
        """
        assert os.path.exists(MODEL_PATH), f"Model file not found! Path: {MODEL_PATH}"

    def test_model_loading(self, shared_session):
        """
        Test: Can the ONNX Runtime load the model without errors?
        We use the fixture here. If fixture worked, session is not None.
        """
        assert shared_session is not None, "Model session could not be created."

    def test_model_metadata(self, shared_session):
        """
        Test: Are the model's input and output definitions correct?
        """

        # Input Control
        inputs = shared_session.get_inputs()
        assert len(inputs) > 0, "The model has no input layer!"

        # Output Control
        outputs = shared_session.get_outputs()
        assert len(outputs) > 0, "The model has no output layer!"

    def test_prediction_flow(self, shared_session):
        """
        Test: Does the model generate predictions when fed with random (dummy) data?
        """
        input_meta = shared_session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        n_features = input_shape[1]

        dummy_input = np.random.rand(1, n_features).astype(np.float32)

        try:
            result = shared_session.run(None, {input_name: dummy_input})

            prediction = result[0]
            assert len(prediction) > 0, "The guess result came back empty."

            assert isinstance(
                prediction[0][0], (np.floating, float)
            ), "The model did not return a numerical value"

        except Exception as e:
            pytest.fail(f"Error during inference: {str(e)}")
