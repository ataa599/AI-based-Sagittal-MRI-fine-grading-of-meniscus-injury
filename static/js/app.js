// Analyze hardcoded images functionality
async function analyzeImages() {
    console.log('analyzeImages function called');
    const statusDiv = document.getElementById('status');
    console.log('statusDiv:', statusDiv);
    
    showStatus('Processing Sagittal.zip images... This may take a few moments.', 'processing');
    
    try {
        console.log('Fetching /infer-hardcoded/');
        const response = await fetch('/infer-hardcoded/', {
            method: 'GET'
        });
        console.log('Response received:', response);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // Store results in sessionStorage
        sessionStorage.setItem('analysisResults', JSON.stringify(data));
        
        showStatus('Analysis complete! Redirecting to results...', 'success');
        
        // Redirect to results page after a short delay
        setTimeout(() => {
            window.location.href = '/results';
        }, 1500);
        
    } catch (error) {
        console.error('Error:', error);
        showStatus(`Error processing images: ${error.message}. Please try again.`, 'error');
    }
}

// Training functionality
async function startTraining() {
    const trainBtn = document.getElementById('trainBtn');
    const trainingStatusDiv = document.getElementById('trainingStatus');
    
    // Confirm before starting training
    const confirmed = confirm(
        'Warning: This will start a comprehensive training pipeline that may take several hours to complete. ' +
        'The existing model will be overwritten. Do you want to proceed?'
    );
    
    if (!confirmed) {
        return;
    }
    
    // Disable button and show status
    trainBtn.disabled = true;
    trainBtn.textContent = 'Training in Progress...';
    showTrainingStatus('Starting training pipeline... This may take several hours. Please do not close this page.', 'processing');
    
    try {
        const response = await fetch('/train/', {
            method: 'GET'
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        showTrainingStatus('Training completed successfully! The model has been updated.', 'success');
        trainBtn.textContent = 'Start Training Pipeline';
        trainBtn.disabled = false;
        
    } catch (error) {
        console.error('Error:', error);
        showTrainingStatus(`Error during training: ${error.message}. Please check the logs and try again.`, 'error');
        trainBtn.textContent = 'Start Training Pipeline';
        trainBtn.disabled = false;
    }
}

// Show status message
function showStatus(message, type) {
    const statusDiv = document.getElementById('status');
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = `status-message show ${type}`;
    }
}

// Show training status message
function showTrainingStatus(message, type) {
    const statusDiv = document.getElementById('trainingStatus');
    if (statusDiv) {
        statusDiv.textContent = message;
        statusDiv.className = `status-message show ${type}`;
    }
}

// Display results on results page
function displayStoredResults() {
    const resultsContainer = document.getElementById('results');
    
    if (!resultsContainer) {
        return; // Not on results page
    }
    
    const storedData = sessionStorage.getItem('analysisResults');
    
    if (!storedData) {
        resultsContainer.innerHTML = `
            <div class="result-card" style="grid-column: 1 / -1;">
                <p>No results available. Please upload images from the <a href="/">home page</a>.</p>
            </div>
        `;
        return;
    }
    
    try {
        const data = JSON.parse(storedData);
        displayResults(data);
    } catch (error) {
        console.error('Error parsing results:', error);
        resultsContainer.innerHTML = `
            <div class="result-card" style="grid-column: 1 / -1;">
                <p>Error loading results. Please try uploading again from the <a href="/">home page</a>.</p>
            </div>
        `;
    }
}

// Display results in grid format
function displayResults(data) {
    const resultsContainer = document.getElementById('results');
    
    const regions = [
        { key: 'posterior_horn_image', title: 'Posterior Horn' },
        { key: 'body_image', title: 'Body' },
        { key: 'anterior_horn_image', title: 'Anterior Horn' }
    ];
    
    let html = '';
    
    regions.forEach(region => {
        const regionData = data[region.key];
        
        if (regionData) {
            html += `
                <div class="result-card">
                    <h3>${regionData.region || region.title}</h3>
                    <img src="data:image/png;base64,${regionData.image_base64}" 
                         alt="${region.title}" 
                         loading="lazy">
                    <div class="result-info">
                        <div>
                            <strong>Predicted Severity:</strong> 
                            ${regionData.predicted_severity}
                        </div>
                        <div>
                            <strong>Confidence:</strong> 
                            ${regionData.confidence}
                        </div>
                    </div>
                </div>
            `;
        }
    });
    
    if (!html) {
        html = `
            <div class="result-card" style="grid-column: 1 / -1;">
                <p>No analysis results available.</p>
            </div>
        `;
    }
    
    resultsContainer.innerHTML = html;
}

// DOMContentLoaded event for results page
document.addEventListener('DOMContentLoaded', () => {
    // Display results if on results page
    displayStoredResults();
});
