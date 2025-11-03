const recordButton = document.getElementById('recordButton');
const statusText = document.getElementById('status');
const transcriptionDiv = document.getElementById('transcription');
const languageSelect = document.getElementById('language');
const processingIndicator = document.getElementById('processingIndicator');

// The URL of our Python Flask backend
const BACKEND_URL = 'http://127.0.0.1:5000/transcribe';

let isRecording = false;
let mediaRecorder;
let audioChunks = [];

function showProcessingIndicator() {
    if (processingIndicator) {
        processingIndicator.style.display = 'block';
        processingIndicator.innerHTML = `
            <div class="processing-spinner">
                <div class="spinner"></div>
                <span>Processing your request...</span>
            </div>
        `;
    }
}

function hideProcessingIndicator() {
    if (processingIndicator) {
        processingIndicator.style.display = 'none';
    }
}

recordButton.addEventListener('click', async () => {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm; codecs=opus' });

            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            };

            mediaRecorder.onstop = sendAudioToServer;
            mediaRecorder.start();
            isRecording = true;
            recordButton.textContent = 'Stop Recording';
            statusText.textContent = 'Status: Recording...';
            transcriptionDiv.textContent = '';
        } catch (error) {
            console.error("Error accessing microphone:", error);
            statusText.textContent = 'Error: Could not access microphone.';
        }
    } else {
        mediaRecorder.stop();
        isRecording = false;
        recordButton.textContent = 'Start Recording';
        statusText.textContent = 'Status: Processing...';
        showProcessingIndicator();
    }
});

async function sendAudioToServer() {
    const audioBlob = new Blob(audioChunks, { type: 'audio/webm; codecs=opus' });
    audioChunks = [];

    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = async () => {
        const base64Audio = reader.result.split(',')[1];
        const selectedLanguage = languageSelect.value;

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ audio: base64Audio, languageCode: selectedLanguage }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            if (data.text) {
                transcriptionDiv.textContent = 'Transcription: ' + data.text;
                statusText.textContent = 'Status: Sending to bot...';

                // Now send the transcription to the bot
                try {
                    const botResponse = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: data.text }),
                    });

                    if (!botResponse.ok) {
                        throw new Error(`Bot error: ${botResponse.statusText}`);
                    }

                    const botData = await botResponse.json();
                    transcriptionDiv.innerHTML += '<br><br><strong>Bot Answer:</strong> ' + botData.answer;
                    statusText.textContent = 'Status: Idle';
                    hideProcessingIndicator();
                } catch (error) {
                    console.error("Error sending to bot:", error);
                    transcriptionDiv.innerHTML += '<br><br><strong>Bot Error:</strong> ' + error.message;
                    statusText.textContent = 'Status: Bot Error';
                    hideProcessingIndicator();
                }
            } else {
                transcriptionDiv.textContent = 'Error: ' + (data.error || 'Unknown error');
                statusText.textContent = 'Status: Error';
                hideProcessingIndicator();
            }
        } catch (error) {
            console.error("Error sending audio to server:", error);
            statusText.textContent = 'Status: Failed to connect to server.';
        }
    };
}