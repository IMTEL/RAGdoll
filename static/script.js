async function testEndpoint(endpoint, method, responseElementId, payload = null, headers = {}) {
    const responseArea = document.getElementById(responseElementId);
    responseArea.textContent = 'Loading...';
    const options = {
        method: method,
        headers: headers
    };
    if (payload && (method === 'POST' || method === 'PUT')) {
        options.body = payload;
    }

    try {
        const response = await fetch(endpoint, options);
        const data = await response.json();
        responseArea.textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
            responseArea.textContent = `Error ${response.status}: ${response.statusText}\n\n${responseArea.textContent}`;
            responseArea.style.color = 'red';
        } else {
            responseArea.style.color = 'green';
        }
    } catch (error) {
        responseArea.textContent = `Fetch Error: ${error}`;
        responseArea.style.color = 'red';
    }
}

function testAskEndpoint() {
    const payload = document.getElementById('ask-payload').value;
    testEndpoint('/ask', 'POST', 'ask-response', payload, { 'Content-Type': 'application/json' });
}

// --- Audio Recording Variables ---
let mediaRecorder;
let audioChunks = [];
let recordedAudioBlob = null;

// Variables for transcribe-specific recording
let transcribeMediaRecorder;
let transcribeAudioChunks = [];
let transcribeAudioBlob = null;

// --- Recording Functions ---
async function startRecording() {
    const startBtn = document.getElementById('start-record-btn');
    const stopBtn = document.getElementById('stop-record-btn');
    const statusIndicator = document.getElementById('recording-status');
    const audioPlayback = document.getElementById('audio-playback');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
            recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Specify WAV type, though browser might use its default
            audioChunks = []; // Reset chunks
            const audioUrl = URL.createObjectURL(recordedAudioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.style.display = 'block'; // Show playback element

            // Clear any file input to prioritize the recording
            document.getElementById('audio-file').value = '';

            // Update UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusIndicator.textContent = '(Recording completed)';
            statusIndicator.style.color = 'green';

            // Clean up the stream tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        audioChunks = []; // Clear previous chunks if any
        recordedAudioBlob = null; // Clear previous blob
        audioPlayback.style.display = 'none'; // Hide playback initially
        audioPlayback.src = ''; // Clear previous source
        mediaRecorder.start();

        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusIndicator.textContent = '(Recording...)';
        statusIndicator.style.color = 'red';

    } catch (err) {
        console.error("Error accessing microphone:", err);
        statusIndicator.textContent = `(Error: ${err.name})`;
        statusIndicator.style.color = 'orange';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        alert(`Could not access microphone: ${err.message}`);
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        // UI updates happen in mediaRecorder.onstop
    }
}

// --- Transcribe Recording Functions ---
async function startTranscribeRecording() {
    const startBtn = document.getElementById('transcribe-start-record-btn');
    const stopBtn = document.getElementById('transcribe-stop-record-btn');
    const statusIndicator = document.getElementById('transcribe-recording-status');
    const audioPlayback = document.getElementById('transcribe-audio-playback');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        transcribeMediaRecorder = new MediaRecorder(stream);

        transcribeMediaRecorder.ondataavailable = event => {
            transcribeAudioChunks.push(event.data);
        };

        transcribeMediaRecorder.onstop = () => {
            transcribeAudioBlob = new Blob(transcribeAudioChunks, { type: 'audio/wav' }); // Specify WAV type
            transcribeAudioChunks = []; // Reset chunks
            const audioUrl = URL.createObjectURL(transcribeAudioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.style.display = 'block'; // Show playback element

            // Clear any file input to prioritize the recording
            document.getElementById('transcribe-audio-file').value = '';

            // Update UI
            startBtn.disabled = false;
            stopBtn.disabled = true;
            statusIndicator.textContent = '(Recording completed)';
            statusIndicator.style.color = 'green';

            // Clean up the stream tracks
            stream.getTracks().forEach(track => track.stop());
        };

        // Start recording
        transcribeAudioChunks = []; // Clear previous chunks if any
        transcribeAudioBlob = null; // Clear previous blob
        audioPlayback.style.display = 'none'; // Hide playback initially
        audioPlayback.src = ''; // Clear previous source
        transcribeMediaRecorder.start();

        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusIndicator.textContent = '(Recording...)';
        statusIndicator.style.color = 'red';

    } catch (err) {
        console.error("Error accessing microphone:", err);
        statusIndicator.textContent = `(Error: ${err.name})`;
        statusIndicator.style.color = 'orange';
        startBtn.disabled = false;
        stopBtn.disabled = true;
        alert(`Could not access microphone: ${err.message}`);
    }
}

function stopTranscribeRecording() {
    if (transcribeMediaRecorder && transcribeMediaRecorder.state === "recording") {
        transcribeMediaRecorder.stop();
        // UI updates happen in transcribeMediaRecorder.onstop
    }
}

async function testTranscribeEndpoint() {
    const responseArea = document.getElementById('transcribe-response');
    responseArea.textContent = 'Loading...';
    const audioFile = document.getElementById('transcribe-audio-file').files[0];
    const language = document.getElementById('transcribe-language').value;

    let audioSource = null;
    let audioFilename = 'audio.wav'; // Default filename

    if (transcribeAudioBlob) {
        audioSource = transcribeAudioBlob;
        audioFilename = 'recording.wav'; // Use specific name for recorded audio
        console.log("Using recorded audio blob for transcription.");
        // Update UI to indicate using recorded audio
        responseArea.textContent = 'Loading... (using recorded audio)';
    } else if (audioFile) {
        audioSource = audioFile;
        audioFilename = audioFile.name;
        console.log("Using uploaded audio file for transcription.");
        // Update UI to indicate using uploaded file
        responseArea.textContent = `Loading... (using uploaded file: ${audioFile.name})`;
    }

    if (!audioSource) {
        responseArea.textContent = 'Please record audio or select an audio file.';
        responseArea.style.color = 'orange';
        return;
    }

    const formData = new FormData();
    // The backend expects the audio file under the key 'audio'
    formData.append('audio', audioSource, audioFilename);
    // Add language if provided
    if (language) {
        formData.append('language', language);
    }

    try {
        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
            // Content-Type is set automatically by browser for FormData
        });
        const data = await response.json();
        responseArea.textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
            responseArea.textContent = `Error ${response.status}: ${response.statusText}\n\n${responseArea.textContent}`;
            responseArea.style.color = 'red';
        } else {
            responseArea.style.color = 'green';
        }
    } catch (error) {
        responseArea.textContent = `Fetch Error: ${error}`;
        responseArea.style.color = 'red';
    }
}

async function testAskTranscribeEndpoint() {
    const responseArea = document.getElementById('ask-transcribe-response');
    responseArea.textContent = 'Loading...';
    const audioFile = document.getElementById('audio-file').files[0];
    const jsonData = document.getElementById('ask-transcribe-payload').value;

    let audioSource = null;
    let audioFilename = 'audio.wav'; // Default filename

    if (recordedAudioBlob) {
        audioSource = recordedAudioBlob;
        audioFilename = 'recording.wav'; // Use specific name for recorded audio
        console.log("Using recorded audio blob.");
        // Update UI to indicate using recorded audio
        responseArea.textContent = 'Loading... (using recorded audio)';
    } else if (audioFile) {
        audioSource = audioFile;
        audioFilename = audioFile.name;
        console.log("Using uploaded audio file.");
        // Update UI to indicate using uploaded file
        responseArea.textContent = `Loading... (using uploaded file: ${audioFile.name})`;
    }

    if (!audioSource) {
        responseArea.textContent = 'Please record audio or select an audio file.';
        responseArea.style.color = 'orange';
        return;
    }
    if (!jsonData) {
        responseArea.textContent = 'Please provide JSON data.';
        responseArea.style.color = 'orange';
        return;
    }

    const formData = new FormData();
    // The backend expects the audio file under the key 'audio'
    formData.append('audio', audioSource, audioFilename);
    // FastAPI expects the JSON part within a form field named 'data'
    formData.append('data', jsonData);

    try {
        const response = await fetch('/askTranscribe', {
            method: 'POST',
            body: formData
            // Content-Type is set automatically by browser for FormData
        });
        const data = await response.json();
        responseArea.textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
            responseArea.textContent = `Error ${response.status}: ${response.statusText}\n\n${responseArea.textContent}`;
            responseArea.style.color = 'red';
        } else {
            responseArea.style.color = 'green';
        }
    } catch (error) {
        responseArea.textContent = `Fetch Error: ${error}`;
        responseArea.style.color = 'red';
    }
}

async function testUploadEndpoint() {
    const responseArea = document.getElementById('upload-response');
    responseArea.textContent = 'Loading...';
    const file = document.getElementById('upload-file').files[0];
    const npc = document.getElementById('upload-npc').value;

    if (!file) {
        responseArea.textContent = 'Please select a file to upload.';
        responseArea.style.color = 'orange';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    // The endpoint expects NPC as a query parameter, not in FormData
    const endpoint = `/upload/?NPC=${encodeURIComponent(npc)}`;

    try {
        const response = await fetch(endpoint, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        responseArea.textContent = JSON.stringify(data, null, 2);
        if (!response.ok) {
            responseArea.textContent = `Error ${response.status}: ${response.statusText}\n\n${responseArea.textContent}`;
            responseArea.style.color = 'red';
        } else {
            responseArea.style.color = 'green';
        }
    } catch (error) {
        responseArea.textContent = `Fetch Error: ${error}`;
        responseArea.style.color = 'red';
    }
}

function testProgressEndpoint(endpoint, payloadElementId, responseElementId) {
    const payload = document.getElementById(payloadElementId).value;
    try {
        // Validate JSON before sending
        JSON.parse(payload);
        testEndpoint(endpoint, 'POST', responseElementId, payload, { 'Content-Type': 'application/json' });
    } catch (e) {
        const responseArea = document.getElementById(responseElementId);
        responseArea.textContent = `Invalid JSON: ${e.message}`;
        responseArea.style.color = 'orange';
    }
}

// Initialize textareas with default JSON if empty
document.addEventListener('DOMContentLoaded', () => {
    const askPayload = document.getElementById('ask-payload');
    if (!askPayload.value) {
        askPayload.value = JSON.stringify({
            progress: [],
            user_actions: ["start"],
            NPC: 1,
            chatLog: [{role: "user", content: "What should I do?"}]
        }, null, 2);
    }

    const askTranscribePayload = document.getElementById('ask-transcribe-payload');
    if (!askTranscribePayload.value) {
        askTranscribePayload.value = JSON.stringify({
            progress: [],
            user_actions: ["start"],
            NPC: 1,
            chatLog: []
        }, null, 2);
    }

    const initTasksPayload = document.getElementById('init-tasks-payload');
    if (!initTasksPayload.value) {
        initTasksPayload.value = JSON.stringify([{
            taskName: "Sample Task",
            description: "Initial task setup",
            status: "pending",
            userId: "init-user",
            subtaskProgress: []
        }], null, 2);
    }

    const updateTaskPayload = document.getElementById('update-task-payload');
    if (!updateTaskPayload.value) {
        updateTaskPayload.value = JSON.stringify({
            taskName: "Sample Task",
            status: "start",
            userId: "test-user123",
            subtaskProgress: []
        }, null, 2);
    }
});