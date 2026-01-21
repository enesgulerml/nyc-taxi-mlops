import os
import sys
import pytest
import onnxruntime as ort
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nyc_taxi_model.onnx")


class TestModelArtifact:
    """
    Model Artifact Tests:
    Tests the existence, loadability, and predictive ability of the trained .onnx model file.
    """

    def test_model_file_exists(self):
        """
        Test: Does the model file physically exist on the disk?
        """
        assert os.path.exists(MODEL_PATH), \
            f"Model file not found! Path: {MODEL_PATH}"

    def test_model_loading(self):
        """
        Test: Can the ONNX Runtime load the model without errors?
        """
        try:
            session = ort.InferenceSession(MODEL_PATH)
            assert session is not None
        except Exception as e:
            pytest.fail(f"An error occurred while loading the model: {str(e)}")

    def test_model_metadata(self):
        """
        Test: Are the model's input and output definitions correct?
        """
        session = ort.InferenceSession(MODEL_PATH)

        # Input Control
        inputs = session.get_inputs()
        assert len(inputs) > 0, "The model has no input layer!"

        # Output Control
        outputs = session.get_outputs()
        assert len(outputs) > 0, "The model has no output layer!"

    def test_prediction_flow(self):
        """
        Test: Does the model generate predictions when fed with random (dummy) data?
        """
        session = ort.InferenceSession(MODEL_PATH)

        # We dynamically retrieve the input name and shape that the model expects.
        input_meta = session.get_inputs()[0]
        input_name = input_meta.name
        input_shape = input_meta.shape

        # Shape usually returns [None, N_Features]. We replace None with 1 (batch size).
        # We get N_Features dynamically (e.g., 6, 8, 10, whatever).
        n_features = input_shape[1]

        # Generate random float data (1 row, N features)
        dummy_input = np.random.rand(1, n_features).astype(np.float32)

        # Predict
        try:
            result = session.run(None, {input_name: dummy_input})

            # Result Check
            prediction = result[0]
            assert len(prediction) > 0, "The guess result came back empty."

            # Is the returned value a number?
            assert isinstance(prediction[0][0], (np.floating, float)), \
                "The model did not return a numerical value"

        except Exception as e:
            pytest.fail(f"Error during inference: {str(e)}")