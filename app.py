import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")




from flask import Flask, request, jsonify
import whisper
import os


# Load the Whisper model
model = whisper.load_model("turbo")

# Initialize the Flask app
app = Flask(__name__)

# Define an endpoint to transcribe audio
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    try:
        # Check if a file is part of the request
        if 'audio' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['audio']

        # Check if a file was actually uploaded
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        print("\033[92m=============> Your request is received and is being processed......\033[0m")

        # Save the uploaded file temporarily
        file_path = os.path.join("temp_audio.mp3")
        file.save(file_path)

        # Perform transcription
        result = model.transcribe(file_path)
        transcription = result.get("text", "")

        # Remove the temporary file
        os.remove(file_path)

        print("\033[92m=============> Your request is complete\033[0m")

        # Return the transcription
        return jsonify({"transcription": transcription})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)