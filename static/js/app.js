// Global variables
let audioContext;
let audioRecorder;
let isRecording = false;
let recordedBlob = null;
let uploadedAudioBlob = null;
let wavesurfer = null;
let recordingStream = null;
let audioBuffer = null;

// DOM elements
const recordButton = document.getElementById('recordButton');
const playButton = document.getElementById('playButton');
const analyzeButton = document.getElementById('analyzeButton');
const audioFileInput = document.getElementById('audioFileInput');
const browseButton = document.getElementById('browseButton');
const waveformContainer = document.getElementById('waveform');
const spectrogramCanvas = document.getElementById('spectrogramCanvas');
const loadingResults = document.getElementById('loadingResults');
const resultsContainer = document.getElementById('resultsContainer');
const statusBar = document.getElementById('statusBar');
const recordIcon = document.getElementById('recordIcon');

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Initialize WaveSurfer
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#3498db',
        progressColor: '#2980b9',
        cursorColor: '#e74c3c',
        barWidth: 2,
        barRadius: 3,
        cursorWidth: 1,
        height: 200,
        responsive: true
    });

    // Set up event listeners
    setupEventListeners();
    
    // Check browser support
    checkBrowserSupport();
});

function checkBrowserSupport() {
    // Check if browser supports required features
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateStatus('Your browser does not support audio recording. Please try a modern browser like Chrome or Firefox.', true);
        recordButton.disabled = true;
        recordButton.title = 'Recording not supported in your browser';
    }
}

function setupEventListeners() {
    // Handle record button click
    recordButton.addEventListener('click', toggleRecording);

    // Handle play button click
    playButton.addEventListener('click', playAudio);

    // Handle analyze button click
    analyzeButton.addEventListener('click', analyzeAudio);

    // Handle file input change
    audioFileInput.addEventListener('change', handleFileSelection);

    // Handle browse button click
    browseButton.addEventListener('click', () => audioFileInput.click());

    // Handle wavesurfer ready event
    wavesurfer.on('ready', function() {
        playButton.disabled = false;
        analyzeButton.disabled = false;
        updateStatus('Audio file loaded and ready for analysis');
    });

    // Handle wavesurfer finished event
    wavesurfer.on('finish', function() {
        playButton.textContent = '▶ Play Audio';
    });
    
    // Handle error events
    wavesurfer.on('error', function(err) {
        console.error('WaveSurfer error:', err);
        updateStatus('Error loading audio file', true);
    });
}

async function toggleRecording() {
    if (!isRecording) {
        // Start recording
        try {
            // Request microphone access
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }

            if (!recordingStream) {
                recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            }

            // Create media recorder
            audioRecorder = new MediaRecorder(recordingStream);
            const audioChunks = [];

            // Handle data available event
            audioRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.push(event.data);
                }
            };

            // Handle recording stop
            audioRecorder.onstop = () => {
                recordedBlob = new Blob(audioChunks, { type: 'audio/wav' });
                uploadedAudioBlob = recordedBlob; // Save for analysis
                const audioUrl = URL.createObjectURL(recordedBlob);
                
                // Update WaveSurfer
                wavesurfer.load(audioUrl);
                
                // Update UI
                recordButton.classList.remove('recording');
                recordButton.textContent = '🎤 Record Audio';
                recordIcon.textContent = '🎤';
                playButton.disabled = false;
                analyzeButton.disabled = false;
                updateStatus('Recording complete. Ready for analysis.');
            };

            // Start recording
            audioRecorder.start();
            isRecording = true;
            
            // Update UI
            recordButton.classList.add('recording');
            recordButton.textContent = '⏹ Stop Recording';
            recordIcon.textContent = '⏺';
            playButton.disabled = true;
            analyzeButton.disabled = true;
            updateStatus('Recording audio...');

        } catch (error) {
            console.error('Error starting recording:', error);
            updateStatus('Error accessing microphone', true);
            showError('Could not access microphone. Please ensure you have granted permission.');
        }
    } else {
        // Stop recording
        if (audioRecorder && audioRecorder.state === 'recording') {
            audioRecorder.stop();
            isRecording = false;
        }
    }
}

function handleFileSelection(event) {
    const file = event.target.files[0];
    if (file && file.type === 'audio/wav') {
        uploadedAudioBlob = file;
        const fileUrl = URL.createObjectURL(file);
        
        // Update UI
        wavesurfer.load(fileUrl);
        updateStatus(`Loaded file: ${file.name}`);
    } else if (file) {
        showError('Please select a WAV audio file');
        audioFileInput.value = '';
    }
}

function playAudio() {
    if (!wavesurfer) return;
    
    if (wavesurfer.isPlaying()) {
        wavesurfer.pause();
        playButton.textContent = '▶ Play Audio';
    } else {
        wavesurfer.play();
        playButton.textContent = '⏸ Pause Audio';
    }
}

async function analyzeAudio() {
    if (!uploadedAudioBlob) {
        showError('Please record or upload an audio file first');
        return;
    }

    // Update UI
    loadingResults.classList.remove('d-none');
    resultsContainer.innerHTML = '';
    analyzeButton.disabled = true;
    updateStatus('Analyzing audio...');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('audio', uploadedAudioBlob);

        // Send to backend for analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to analyze audio');
        }

        // Process and display results
        const results = await response.json();
        displayResults(results);
        generateSpectrogram(); // Generate spectrogram visualization
        
    } catch (error) {
        console.error('Error analyzing audio:', error);
        updateStatus('Error analyzing audio', true);
        showError(`An error occurred during analysis: ${error.message}`);
    } finally {
        loadingResults.classList.add('d-none');
        analyzeButton.disabled = false;
    }
}

function displayResults(results) {
    // Clear previous results
    resultsContainer.innerHTML = '';
    
    if (!results || !results.detections || results.detections.length === 0) {
        resultsContainer.innerHTML = '<p class="text-center text-muted">No pests detected in this audio sample.</p>';
        return;
    }

    // Get the result template
    const template = document.getElementById('resultTemplate');
    const resultElement = template.content.cloneNode(true);
    
    // Get the primary detection
    const primaryDetection = results.detections[0];
    
    // Populate primary detection data
    resultElement.querySelector('.detected-pest').textContent = primaryDetection.pest;
    
    // Set confidence bar
    const confidenceBar = resultElement.querySelector('.progress-bar');
    confidenceBar.style.width = `${primaryDetection.confidence * 100}%`;
    confidenceBar.textContent = `${(primaryDetection.confidence * 100).toFixed(1)}%`;
    
    if (primaryDetection.confidence < 0.5) {
        confidenceBar.classList.add('bg-warning');
    } else {
        confidenceBar.classList.add('bg-success');
    }
    
    // Set confidence text
    const confidenceText = resultElement.querySelector('.confidence-text');
    confidenceText.textContent = `Confidence: ${(primaryDetection.confidence * 100).toFixed(1)}%`;
    
    // Add warnings if needed
    if (primaryDetection.confidence < 0.3) {
        const warningEl = document.createElement('p');
        warningEl.className = 'warning-text';
        warningEl.textContent = '⚠️ WARNING: Low confidence detection';
        confidenceText.appendChild(warningEl);
    }
    
    // Set pest details
    resultElement.querySelector('.scientific-name').textContent = primaryDetection.scientific_name;
    resultElement.querySelector('.description').textContent = primaryDetection.description;
    resultElement.querySelector('.impact').textContent = primaryDetection.impact;
    resultElement.querySelector('.dominant-freq').textContent = `${primaryDetection.dominant_freq} Hz`;
    resultElement.querySelector('.frequency-range').textContent = primaryDetection.frequency_range;
    resultElement.querySelector('.sound-pattern').textContent = primaryDetection.sound_pattern;
    
    // Add alternative detections
    const alternativesContainer = resultElement.querySelector('.alternative-detections');
    
    if (results.detections.length > 1) {
        results.detections.slice(1).forEach((detection, index) => {
            const altItem = document.createElement('div');
            altItem.className = 'alternative-item';
            
            altItem.innerHTML = `
                <p><strong>${index + 2}. ${detection.pest}</strong> (Confidence: ${(detection.confidence * 100).toFixed(1)}%)</p>
                <p><strong>Scientific Name:</strong> ${detection.scientific_name}</p>
            `;
            
            alternativesContainer.appendChild(altItem);
        });
    } else {
        alternativesContainer.innerHTML = '<p>No alternative detections found.</p>';
    }
    
    // Add to results container
    resultsContainer.appendChild(resultElement);
    updateStatus(`Analysis complete. Detected: ${primaryDetection.pest}`);
}

function generateSpectrogram() {
    // Get the canvas and its context
    const canvas = document.getElementById('spectrogramCanvas');
    const ctx = canvas.getContext('2d');
    
    // Set canvas dimensions
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Create a gradient for the spectrogram
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0.0, '#1a5276');
    gradient.addColorStop(0.3, '#3498db');
    gradient.addColorStop(0.6, '#2ecc71');
    gradient.addColorStop(1.0, '#f1c40f');
    
    ctx.fillStyle = gradient;
    
    // Generate a simulated spectrogram (in a real app, this would use FFT data)
    for (let x = 0; x < canvas.width; x += 2) {
        // Create a pattern that looks somewhat like an insect sound
        let y = 0;
        
        // Create some frequency bands
        const baseHeight = Math.sin(x * 0.05) * 20 + 40;
        const bandHeight1 = baseHeight + Math.sin(x * 0.1) * 10;
        const bandHeight2 = baseHeight * 0.6 + Math.sin(x * 0.2) * 15;
        const bandHeight3 = baseHeight * 0.3 + Math.sin(x * 0.3) * 5;
        
        // Draw main frequency components
        ctx.fillRect(x, canvas.height - bandHeight1, 2, bandHeight1);
        ctx.fillRect(x, canvas.height - bandHeight2 - 60, 2, bandHeight2);
        ctx.fillRect(x, canvas.height - bandHeight3 - 100, 2, bandHeight3);
        
        // Add some random noise
        if (Math.random() > 0.7) {
            const noiseHeight = Math.random() * 30;
            const noisePos = Math.random() * (canvas.height - 30);
            ctx.fillRect(x, noisePos, 2, noiseHeight);
        }
    }
}

function updateStatus(message, isError = false) {
    statusBar.textContent = message;
    
    if (isError) {
        statusBar.classList.add('text-danger');
        statusBar.classList.remove('text-white');
    } else {
        statusBar.classList.add('text-white');
        statusBar.classList.remove('text-danger');
    }
}

function showError(message) {
    // Create a bootstrap alert
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.role = 'alert';
    
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to the page
    const container = document.querySelector('.container');
    container.insertBefore(alertDiv, container.firstChild);
    
    // Update status
    updateStatus(message, true);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(alertDiv);
        bsAlert.close();
    }, 5000);
}
