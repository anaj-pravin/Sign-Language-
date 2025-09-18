// === Global Variables ===
let isLoading = false;

// === Socket.IO Connection ===
const protocol = location.protocol; // e.g., "http:" or "https:"
const hostname = location.hostname; // e.g., "localhost" or "example.com"
const port = location.port ? `:${location.port}` : ''; // e.g., ":5000" or ""
const socket = io.connect(`${protocol}//${hostname}${port}`);

// Log WebSocket connection status
socket.on('connect', () => {
    console.log('Connected to WebSocket');
});

socket.on('connect_error', (error) => {
    console.error('WebSocket connection error:', error);
    alert('Failed to connect to the server. Please refresh the page.');
});

socket.on('disconnect', () => {
    console.warn('Disconnected from WebSocket. Attempting to reconnect...');
    alert('Lost connection to the server. Please refresh the page.');
});

socket.on('update_text', (data) => {
    console.log('Received update_text event:', data);  // Debug log
    const textArea = document.getElementById("recognized-text");
    if (textArea) {
        textArea.value = data.recognized_text || "";  // Update textarea directly
        adjustTextHeight(textArea);
        console.log('Textarea updated to:', textArea.value);  // Confirm update
    } else {
        console.error('Textarea element not found!');
    }
});

// === Utility Functions ===
function adjustTextHeight(element) {
    element.style.height = 'auto'; // Reset height to auto to calculate scrollHeight
    const maxHeight = 200; // Match the max-height in CSS
    const newHeight = Math.min(element.scrollHeight, maxHeight);
    element.style.height = `${newHeight}px`;
}

function showLoading() {
    isLoading = true;
    document.getElementById('loading').style.display = 'block';
}

function hideLoading() {
    isLoading = false;
    document.getElementById('loading').style.display = 'none';
}

function handleVideoError() {
    console.error('Failed to load video feed.');
    document.getElementById('video-error').style.display = 'block';
}

// === Feature Functions ===
function speakText() {
    showLoading();
    const text = document.getElementById("recognized-text").value;
    if (text) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.onend = () => hideLoading();
        utterance.onerror = (error) => {
            console.error('Text-to-speech error:', error);
            alert('An error occurred during text-to-speech. Please try again.');
            hideLoading();
        };
        window.speechSynthesis.speak(utterance);
    } else {
        alert('No text to speak.');
        hideLoading();
    }
}

function copyText() {
    showLoading();
    const text = document.getElementById("recognized-text").value;
    if (!text) {
        alert('No text to copy.');
        hideLoading();
        return;
    }

    navigator.clipboard.writeText(text)
        .then(() => {
            alert('Text copied to clipboard.');
            hideLoading();
        })
        .catch((err) => {
            console.error('Clipboard copy error:', err);
            const textarea = document.createElement("textarea");
            textarea.value = text;
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand("copy");
            document.body.removeChild(textarea);
            alert('Text copied to clipboard (fallback method).');
            hideLoading();
        });
}

function clearText() {
    showLoading();
    fetch('/clear_text', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const textContent = document.getElementById("recognized-text");
            textContent.value = "";
            adjustTextHeight(textContent);
            hideLoading();
        })
        .catch(error => {
            console.error('Error in clearText:', error);
            alert('An error occurred while clearing the text. Please try again.');
            hideLoading();
        });
}

function backspaceText() {
    showLoading();
    fetch('/backspace', { method: 'POST' })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
        })
        .catch(error => {
            console.error('Error in backspaceText:', error);
            alert('An error occurred while processing backspace. Please try again.');
            hideLoading();
        });
}

function confirmSave() {
    const text = document.getElementById("recognized-text").value;
    const filename = document.getElementById("filename").value.trim() || "sign_language_output";
    if (!text) {
        alert("No text to save.");
        hideLoading();
        return;
    }

    showLoading();
    fetch('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text, filename: filename })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        alert(`Text saved as ${data.filename}`);
        hideLoading();
    })
    .catch(error => {
        console.error('Error in confirmSave:', error);
        alert('An error occurred while saving the text. Please try again.');
        hideLoading();
    });
}