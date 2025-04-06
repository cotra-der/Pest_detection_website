document.addEventListener('DOMContentLoaded', function() {
    const audioForm = document.getElementById('audioForm');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');
    const audioPlayer = document.getElementById('audioPlayer');
    const predictionResult = document.getElementById('predictionResult');
    const errorMessage = document.getElementById('errorMessage');

    if (audioForm) {
        audioForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Show loading state
            if (loadingDiv) loadingDiv.style.display = 'block';
            if (resultDiv) resultDiv.style.display = 'none';
            if (errorMessage) errorMessage.style.display = 'none';
            
            try {
                const formData = new FormData(this);
                console.log("Submitting form data...");
                
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                console.log("Received response:", response.status);
                
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error ${response.status}`);
                }
                
                const result = await response.json();
                console.log("Analysis result:", result);
                
                // Display results
                if (audioPlayer) {
                    audioPlayer.src = result.audio_file;
                    audioPlayer.style.display = 'block';
                }
                
                if (predictionResult) {
                    predictionResult.innerHTML = `
                        <h3>Detection Results:</h3>
                        <p class="prediction-main">
                            <strong>Identified pest:</strong> ${result.prediction.class}<br>
                            <strong>Confidence:</strong> ${(result.prediction.confidence * 100).toFixed(2)}%
                        </p>
                        
                        <h4>Top 3 Predictions:</h4>
                        <ul class="prediction-list">
                            ${result.top_3.map(item => `
                                <li>${item.class}: ${(item.confidence * 100).toFixed(2)}%</li>
                            `).join('')}
                        </ul>
                    `;
                }
                
                // Show results
                if (resultDiv) resultDiv.style.display = 'block';
            } catch (error) {
                console.error('Error:', error);
                if (errorMessage) {
                    errorMessage.textContent = 'Analysis failed: ' + error.message;
                    errorMessage.style.display = 'block';
                }
            } finally {
                // Hide loading state
                if (loadingDiv) loadingDiv.style.display = 'none';
            }
        });
    }
});